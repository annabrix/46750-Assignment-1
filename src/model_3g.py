"""Optimisation model of a single flexible consumer, implemented with gurobipy.

The class below separates the three steps you will repeat for every question:

    model = FlexibleConsumerModel(data)   # 1. hand over the input data
    model.build()                         # 2. declare variables, objective, constraints
    results = model.solve()               # 3. optimise and collect primal AND dual values

``build()`` is the only method you need to complete for Question 1; the other questions
are variations of it (a different objective, an extra constraint). Copy this file or
subclass ``FlexibleConsumerModel`` and override ``build()`` to keep one model per question.

Two conventions make the dual variables easy to read out afterwards:

* Every constraint family is stored in ``self.con`` under a descriptive name, e.g.
  ``self.con["balance"] = self.m.addConstrs(...)``. ``solve()`` then returns the dual value
  (shadow price, Gurobi attribute ``Pi``) of every constraint in ``self.con`` automatically.
* Bounds that you want a dual for must be written as explicit constraints (``addConstr``),
  not as variable bounds (``lb=``/``ub=``). Gurobi reports the sensitivity of a variable
  bound in the reduced cost (``RC``), not in ``Pi``.
* Duals of quadratic constraints (``m.addQConstr``) are read from ``QCPi`` and require ``QCPDual = 1`` (set below).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from .data_loader import InputData


@dataclass
class Results:
    """Primal and dual solution of one model run."""

    question: str
    status: str
    objective: float
    hourly: pd.DataFrame                   # one row per hour: variables, prices, hourly duals
    duals: dict[str, float] = field(default_factory=dict)   # duals of non-hourly constraints
    meta: dict = field(default_factory=dict)                 # anything else worth keeping (scenario name, ...)

    def save(self, folder: Path | str, tag: str = "") -> None:
        """Write ``hourly`` to CSV and the scalar values to a small text file."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        stem = f"{self.question}{'_' + tag if tag else ''}"
        self.hourly.to_csv(folder / f"{stem}_hourly.csv", index_label="hour")
        with open(folder / f"{stem}_summary.txt", "w", encoding="utf-8") as f:
            f.write(f"status    : {self.status}\nobjective : {self.objective:.4f} DKK\n")
            for k, v in self.duals.items():
                f.write(f"dual[{k}] : {v:.4f}\n")

    def __str__(self) -> str:
        cols = [c for c in self.hourly.columns if not c.startswith("dual_")]
        return (
            f"status: {self.status} | objective: {self.objective:.2f} DKK\n"
            f"daily totals (kWh): " + ", ".join(f"{c}={self.hourly[c].sum():.1f}" for c in cols if c in ("import", "export", "load", "pv"))
            + (f"\nduals: {self.duals}" if self.duals else "")
        )


def _dual(c) -> float:
    """Dual value of a linear (``Pi``) or quadratic (``QCPi``, requires QCPDual=1) constraint."""
    return c.QCPi if isinstance(c, gp.QConstr) else c.Pi


class FlexibleConsumerModel:
    """Consumption problem of one consumer over a 24-hour horizon (Question 1); extend it for Questions 2 and 3."""

    def __init__(self, data: InputData, name: str = "flexible_consumer", verbose: bool = False):
        self.data = data
        self.T = range(data.n_hours)
        self.m = gp.Model(name)
        self.m.Params.OutputFlag = 1 if verbose else 0
        self.m.Params.QCPDual = 1          # only relevant if you add a quadratic constraint (none is needed in Assignment 1)
        self.var: dict[str, gp.tupledict | gp.Var] = {}   # decision variables by name
        self.con: dict[str, gp.tupledict | gp.Constr] = {}  # constraints by name (duals read from here)

    # ------------------------------------------------------------------ 2. build
   
    def build(self, terminal_value_DKK_per_kWh=0.0) -> "FlexibleConsumerModel":
        """
        Q3 with battery storage.

        Includes quadratic disutility, minimum daily energy,
        battery charging/discharging, SoC dynamics and
        a terminal battery energy value.
        """

        d, m, T = self.data, self.m, self.T

        # --------------------------------------------------
        # Input validation
        # --------------------------------------------------

        if d.reference_load is None:
            raise ValueError("Reference load is required.")

        if d.quadratic_disutility is None:
            raise ValueError("Quadratic disutility is required.")

        battery_parameters = [
            d.battery_capacity_kWh,
            d.battery_max_charge_kW,
            d.battery_max_discharge_kW,
            d.battery_charging_efficiency,
            d.battery_discharging_efficiency,
            d.battery_initial_soc_kWh,
        ]

        if any(x is None for x in battery_parameters):
            raise ValueError("Battery parameters are missing.")

        if not (0 < d.battery_charging_efficiency <= 1):
            raise ValueError("Invalid charging efficiency.")

        if not (0 < d.battery_discharging_efficiency <= 1):
            raise ValueError("Invalid discharging efficiency.")

        if not (0 <= d.battery_initial_soc_kWh <= d.battery_capacity_kWh):
            raise ValueError("Initial SoC is outside battery capacity.")

        if not T:
            raise ValueError("The optimization horizon is empty.")

        # The repository uses hourly data (1-hour intervals).
        dt = 1.0

        # Effective electricity prices
        p_imp = d.energy_price + d.import_tariff
        p_exp = d.energy_price - d.export_tariff

        # --------------------------------------------------
        # Decision variables
        # --------------------------------------------------

        # All bounds are explicit constraints for dual extraction.

        for name in ["import", "export", "load", "pv",
                    "charge", "discharge", "soc"]:

            self.var[name] = m.addVars(
                T,
                lb=-GRB.INFINITY,
                vtype=GRB.CONTINUOUS,
                name=name
            )

        p_import = self.var["import"]
        p_export = self.var["export"]
        load = self.var["load"]
        pv = self.var["pv"]

        charge = self.var["charge"]
        discharge = self.var["discharge"]
        soc = self.var["soc"]

        # --------------------------------------------------
        # Objective
        # --------------------------------------------------

        last_t = T[-1]

        m.setObjective(
            gp.quicksum(
                d.quadratic_disutility
                * (load[t] - d.reference_load[t]) ** 2

                + d.pv_marginal_cost * pv[t]

                + p_imp[t] * p_import[t]

                - p_exp[t] * p_export[t]

                for t in T
            )

            - terminal_value_DKK_per_kWh
            * (soc[last_t] - d.battery_initial_soc_kWh),

            GRB.MINIMIZE
        )

        # --------------------------------------------------
        # Hourly power balance
        # PV + import + discharge = load + export + charge
        # --------------------------------------------------

        self.con["balance"] = m.addConstrs(
            (
                pv[t] + p_import[t] + discharge[t]
                == load[t] + p_export[t] + charge[t]

                for t in T
            ),
            name="balance"
        )

        # --------------------------------------------------
        # Load bounds
        # --------------------------------------------------

        self.con["load_min"] = m.addConstrs(
            (
                load[t] >= d.load_min_kWh
                for t in T
            ),
            name="load_min"
        )

        self.con["load_max"] = m.addConstrs(
            (
                load[t] <= d.load_max_kWh
                for t in T
            ),
            name="load_max"
        )

        # --------------------------------------------------
        # PV bounds
        # --------------------------------------------------

        self.con["pv_min"] = m.addConstrs(
            (
                pv[t] >= 0
                for t in T
            ),
            name="pv_min"
        )

        self.con["pv_max"] = m.addConstrs(
            (
                pv[t] <= d.pv_available[t]
                for t in T
            ),
            name="pv_max"
        )

        # --------------------------------------------------
        # Grid non-negativity
        # --------------------------------------------------

        self.con["import_min"] = m.addConstrs(
            (
                p_import[t] >= 0
                for t in T
            ),
            name="import_min"
        )

        self.con["export_min"] = m.addConstrs(
            (
                p_export[t] >= 0
                for t in T
            ),
            name="export_min"
        )

        # --------------------------------------------------
        # Minimum daily energy
        # --------------------------------------------------

        if d.min_daily_energy_kWh is not None:
            self.con["min_daily_energy"] = m.addConstr(
                gp.quicksum(load[t] for t in T)
                >= d.min_daily_energy_kWh,
                name="min_daily_energy"
            )

        # --------------------------------------------------
        # Battery charging and discharging bounds
        # --------------------------------------------------

        self.con["charge_min"] = m.addConstrs(
            (
                charge[t] >= 0
                for t in T
            ),
            name="charge_min"
        )

        self.con["charge_max"] = m.addConstrs(
            (
                charge[t] <= d.battery_max_charge_kW
                for t in T
            ),
            name="charge_max"
        )

        self.con["discharge_min"] = m.addConstrs(
            (
                discharge[t] >= 0
                for t in T
            ),
            name="discharge_min"
        )

        self.con["discharge_max"] = m.addConstrs(
            (
                discharge[t] <= d.battery_max_discharge_kW
                for t in T
            ),
            name="discharge_max"
        )

        # --------------------------------------------------
        # State of charge bounds
        # --------------------------------------------------

        self.con["soc_min"] = m.addConstrs(
            (
                soc[t] >= 0
                for t in T
            ),
            name="soc_min"
        )

        self.con["soc_max"] = m.addConstrs(
            (
                soc[t] <= d.battery_capacity_kWh
                for t in T
            ),
            name="soc_max"
        )

        # --------------------------------------------------
        # Battery state of charge dynamics
        # --------------------------------------------------

        eta_ch = d.battery_charging_efficiency
        eta_dis = d.battery_discharging_efficiency

        # First hour: start from known initial SoC
        self.con["soc_initial"] = m.addConstr(
            soc[T[0]]
            == d.battery_initial_soc_kWh
            + eta_ch * charge[T[0]] * dt
            - discharge[T[0]] * dt / eta_dis,
            name="soc_initial"
        )

        # Remaining hours
        self.con["soc_balance"] = m.addConstrs(
            (
                soc[t]
                == soc[t - 1]
                + eta_ch * charge[t] * dt
                - discharge[t] * dt / eta_dis

                for t in T
                if t > T[0]
            ),
            name="soc_balance"
        )

        m.update()

        return self


    # ------------------------------------------------------------------ 3. solve
    def solve(self) -> Results:
        """Optimise and return primal values, objective and dual values."""
        m = self.m
        m.update()
        if m.NumConstrs == 0 and m.NumQConstrs == 0:
            raise NotImplementedError(
                "The model has no constraints: complete FlexibleConsumerModel.build() in src/model.py first."
            )
        m.optimize()
        status = _status_name(m.Status)
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Optimisation ended with status {status}. Check the model (m.computeIIS() helps for infeasibility).")
        return self._extract_results(status)

    # --------------------------------------------------------------- extraction
    def _extract_results(self, status: str) -> Results:
        d, T = self.data, list(self.T)
        hourly = pd.DataFrame(index=pd.Index(T, name="hour"))
        hourly["price"] = d.energy_price
        hourly["pv_available"] = d.pv_available
        if d.reference_load is not None:
            hourly["reference_load"] = d.reference_load

        # Primal values: every hourly variable family in self.var becomes a column
        for name, v in self.var.items():
            if isinstance(v, gp.tupledict):
                hourly[name] = [v[t].X for t in T]
        scalars = {name: v.X for name, v in self.var.items() if isinstance(v, gp.Var)}

        # Dual values: every constraint family in self.con becomes a 'dual_<name>' column or scalar
        duals: dict[str, float] = {}
        for name, c in self.con.items():
            try:
                if isinstance(c, gp.tupledict):

                    hourly[f"dual_{name}"] = [
                        _dual(c[t]) if t in c else float("nan")
                        for t in T
                    ]

                else:
                    duals[name] = _dual(c)

            except (AttributeError, gp.GurobiError):
                pass

        return Results(
            question=d.question,
            status=status,
            objective=self.m.ObjVal,
            hourly=hourly,
            duals=duals,
            meta={"scalar_variables": scalars},
        )


_STATUS = {
    GRB.OPTIMAL: "OPTIMAL", GRB.INFEASIBLE: "INFEASIBLE", GRB.UNBOUNDED: "UNBOUNDED",
    GRB.INF_OR_UNBD: "INF_OR_UNBD", GRB.TIME_LIMIT: "TIME_LIMIT", GRB.SUBOPTIMAL: "SUBOPTIMAL",
    GRB.NUMERIC: "NUMERIC", GRB.INTERRUPTED: "INTERRUPTED",
}


def _status_name(code: int) -> str:
    return _STATUS.get(code, f"STATUS_{code}")
