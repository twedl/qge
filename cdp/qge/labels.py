"""Human-readable labels for sectors, regions, and quarters.

Used by the reporting layer to turn naked numpy arrays into pandas
DataFrames indexed by sector / region / quarter names. The labels are
also imported by ``scripts/convert_cdp_txt.py`` so the parquet inputs
are labeled identically.
"""

from __future__ import annotations


# 22 productive sectors in the CDP order (CDP Readme §1).
SECTORS: tuple[str, ...] = (
    "Food, Beverage, Tobacco",
    "Textile, Apparel, Leather",
    "Wood, Paper, Printing",
    "Petroleum and Coal",
    "Chemical",
    "Plastics and Rubber",
    "Nonmetallic Mineral Products",
    "Primary Metal and Fabricated Metal",
    "Machinery",
    "Computer, Electronic, Electrical",
    "Transportation Equipment",
    "Furniture and Miscellaneous Manufacturing",
    "Wholesale and Retail Trade",
    "Construction",
    "Transport Services",
    "Information Services",
    "Finance and Insurance",
    "Real Estate",
    "Education",
    "Health Care",
    "Accommodation and Food Services",
    "Other Services",
)
N_TRADABLES = 13                                       # sectors 0..12 — tradables

# 23 labor markets per US state: non-employment + 22 productive.
NONEMPLOYMENT_LABEL = "Non-employment"
LABOR_MARKETS: tuple[str, ...] = (NONEMPLOYMENT_LABEL,) + SECTORS

# 50 US states in CDP alphabetical order (Virginia merged with DC).
US_STATES: tuple[str, ...] = (
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho",
    "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana",
    "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "New York",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
    "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia and DC",
    "Washington", "West Virginia", "Wisconsin", "Wyoming",
)

# 37 foreign countries in the CDP-Readme order (note: not pure alphabetical).
COUNTRIES: tuple[str, ...] = (
    "Australia", "Austria", "Belgium", "Bulgaria", "Brazil", "Canada",
    "China", "Cyprus", "Czech Republic", "Denmark", "Estonia", "Finland",
    "France", "Germany", "Greece", "Hungary", "India", "Indonesia",
    "Italy", "Ireland", "Japan", "Lithuania", "Mexico", "Netherlands",
    "Poland", "Portugal", "Romania", "Russia", "Spain", "Slovak Republic",
    "Slovenia", "South Korea", "Sweden", "Taiwan", "Turkey",
    "United Kingdom", "Rest of World",
)

REGIONS: tuple[str, ...] = US_STATES + COUNTRIES        # 87 total
IO_BLOCKS: tuple[str, ...] = ("United States",) + COUNTRIES   # 38: US shared block + foreign


def quarter_labels(start_year: int = 2000, n_quarters: int = 200) -> tuple[str, ...]:
    """Return ``("2000Q1", "2000Q2", …)`` of length ``n_quarters``.

    CDP's time axis starts at 2000Q1 and runs ``n_quarters`` periods.
    Phase 2a has 29 quarters; Phase 2b has 29; Phase 2c/2d have 200;
    series_mu in Phase 2d has 220 transitions.
    """
    out = []
    for i in range(n_quarters):
        year = start_year + i // 4
        q = (i % 4) + 1
        out.append(f"{year}Q{q}")
    return tuple(out)
