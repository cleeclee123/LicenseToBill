from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Annotated

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set(style="whitegrid")


def plot_bond(
    ytm: float,
    coupon: float,
    years_to_maturity: int,
    calc_bond_duration: Callable[
        [float, float, int, Optional[float], float, int, bool, bool],
        Tuple[float, float],
    ],
    calc_bond_price_from_ytm: Callable[
        [float, float, int, Optional[float], float, int, bool, bool],
        Tuple[float, float],
    ],
    curr_bond_price: Optional[float] = None,
    face_amount: Optional[float] = 100,
    freq: Optional[int] = 2,
    modified: Optional[bool] = False,
    calc_dirty: Optional[bool] = False,
    plot_title: Optional[str] = None,
    plot_size: Optional[Tuple[int, int]] = None,
):
    duration, bond_price = calc_bond_duration(
        ytm=ytm,
        coupon=coupon,
        years_to_maturity=years_to_maturity,
        curr_bond_price=curr_bond_price,
        face_amount=face_amount,
        freq=freq,
        modified=modified,
        calc_dirty=calc_dirty,
    )

    ytm /= 100
    ytms = np.linspace(0.10 * ytm, 2 * ytm, 100)
    prices = [
        calc_bond_price_from_ytm(
            ytm=y * 100,
            coupon=coupon,
            years_to_maturity=years_to_maturity,
            freq=freq,
            face_amount=face_amount,
            calc_dirty=calc_dirty,
        )
        for y in ytms
    ]
    plt.figure(figsize=plot_size or (12, 6))
    plt.plot(ytm * 100, bond_price, "ro", label="Current Price")
    for curr_ytm in ytms:
        diff = (curr_ytm - ytm) * 100
        plt.plot(
            ytm * 100 - diff,
            bond_price + (bond_price * (diff * (duration / 100))),
            "r.",
        )
    plt.plot(ytms * 100, prices, linewidth=2.5, label="Convexity")

    plt.title(plot_title or "Bond Price vs. YTM")
    plt.xlabel("Yield to Maturity (%)")
    plt.ylabel("Price")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()


@dataclass
class BondCharacteristics:
    ytm: float
    coupon: float
    years_to_maturity: float
    freq: int
    label: str
    face_amount: Optional[float] = 100
    calc_dirty: Optional[bool] = False


def plot_bonds(
    bonds: List[BondCharacteristics],
    calc_bond_duration: Callable[
        [float, float, int, Optional[float], float, int, bool, bool],
        Tuple[float, float],
    ],
    calc_bond_price_from_ytm: Callable[
        [float, float, int, Optional[float], float, int, bool, bool],
        Tuple[float, float],
    ],
    modified_duration=True,
    plot_title: Optional[str] = None,
    plot_size: Optional[Tuple[int, int]] = None,
    plot_current_pricing=True,
    plot_current_pricing_line=True,
    plot_duration=True,
    plot_convexity=True,
    verbose=False,
    x_range: Optional[Annotated[List[int], 2]] = None,
):
    plt.figure(figsize=plot_size or (12, 6))
    for bond in bonds:
        duration, curr_bond_price = calc_bond_duration(
            ytm=bond.ytm,
            coupon=bond.coupon,
            years_to_maturity=bond.years_to_maturity,
            freq=bond.freq,
            face_amount=bond.face_amount,
            modified=modified_duration,
            calc_dirty=bond.calc_dirty,
        )

        ytm = bond.ytm
        ytm /= 100
        if x_range:
            ytms = np.linspace(x_range[0] / 100, x_range[1] / 100, 100)
        else:
            ytms = np.linspace(0.10 * ytm, 2 * ytm, 100)

        prices = [
            calc_bond_price_from_ytm(
                ytm=y * 100,
                coupon=bond.coupon,
                years_to_maturity=bond.years_to_maturity,
                freq=bond.freq,
                face_amount=bond.face_amount,
                calc_dirty=bond.calc_dirty,
            )
            for y in ytms
        ]

        if plot_convexity:
            convexity_plot = plt.plot(
                ytms * 100,
                prices,
                linewidth=2,
                label=f"{bond.label} Convexity" if verbose else None,
            )

        if plot_current_pricing:
            current_pricing_plot = plt.plot(
                ytm * 100,
                curr_bond_price,
                "o",
                label=f"{bond.label} Current Price: {round(curr_bond_price, 5)}",
                color=convexity_plot[0].get_color() if plot_convexity else None,
            )

        if plot_current_pricing_line:
            plt.axhline(
                y=curr_bond_price,
                color=current_pricing_plot[0].get_color(),
                linestyle="--",
                linewidth=1,
                # label="Current Bond Price Level",
            )

        if plot_duration:
            for curr_ytm in ytms:
                diff = (curr_ytm - ytm) * 100
                plt.plot(
                    ytm * 100 - diff,
                    curr_bond_price + (curr_bond_price * (diff * (duration / 100))),
                    ".",
                    color=convexity_plot[0].get_color() if plot_convexity else None,
                )

    plt.title(plot_title or "Bond Prices vs. YTM")
    plt.xlabel("Yield to Maturity (%)")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
