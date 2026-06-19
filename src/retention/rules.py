"""Actionable features for the retention planner (Stage 6).

This is the catalog of features the team can realistically act on, each paired with
a retention action and what a "better" value looks like. The planner only turns one
of these into a recommendation when the model flags the feature as raising the
customer's risk AND similar retained customers commonly have the better value.

Protected, personal, or non-changeable features (gender, senior-citizen status,
partner/dependents, customer ID) are deliberately left out. Tenure cannot be
changed either, but a short tenure can trigger an onboarding/loyalty action.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ActionableFeature:
    """One feature the retention team can act on.

    feature: the original column name.
    kind: "categorical" or "numeric".
    action: the retention action to recommend.
    actionability: 0-1 weight for how realistically actionable the feature is.
    better_values: for categorical features, the favorable values to move toward.
    direction: for numeric features, "lower" (e.g. charges) or "newer" (tenure).
    alternative_label: plain wording for the suggested alternative.
    """

    feature: str
    kind: str
    action: str
    actionability: float = 1.0
    better_values: frozenset = field(default_factory=frozenset)
    direction: str = ""
    alternative_label: str = ""


ACTIONABLE_FEATURES = [
    ActionableFeature(
        feature="Contract",
        kind="categorical",
        action="Offer an incentive to move to a longer-term contract.",
        actionability=1.0,
        better_values=frozenset({"One year", "Two year"}),
        alternative_label="a one- or two-year contract",
    ),
    ActionableFeature(
        feature="MonthlyCharges",
        kind="numeric",
        action="Recommend a lower-cost package or a temporary discount.",
        actionability=1.0,
        direction="lower",
        alternative_label="a lower monthly charge",
    ),
    ActionableFeature(
        feature="TechSupport",
        kind="categorical",
        action="Offer a free or discounted technical-support trial.",
        actionability=0.95,
        better_values=frozenset({"Yes"}),
        alternative_label="technical support",
    ),
    ActionableFeature(
        feature="OnlineSecurity",
        kind="categorical",
        action="Offer an online-security add-on trial.",
        actionability=0.95,
        better_values=frozenset({"Yes"}),
        alternative_label="online security",
    ),
    ActionableFeature(
        feature="OnlineBackup",
        kind="categorical",
        action="Offer an online-backup add-on trial.",
        actionability=0.9,
        better_values=frozenset({"Yes"}),
        alternative_label="online backup",
    ),
    ActionableFeature(
        feature="DeviceProtection",
        kind="categorical",
        action="Offer a device-protection add-on trial.",
        actionability=0.9,
        better_values=frozenset({"Yes"}),
        alternative_label="device protection",
    ),
    ActionableFeature(
        feature="PaymentMethod",
        kind="categorical",
        action="Suggest moving to an automatic payment method.",
        actionability=0.9,
        better_values=frozenset(
            {"Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"}
        ),
        alternative_label="an automatic payment method",
    ),
    ActionableFeature(
        feature="InternetService",
        kind="categorical",
        action="Review the internet plan and its value.",
        actionability=0.7,
        better_values=frozenset({"DSL", "No"}),
        alternative_label="the plan similar retained customers use",
    ),
    ActionableFeature(
        feature="PaperlessBilling",
        kind="categorical",
        action="Offer to switch the customer off paperless billing.",
        actionability=0.6,
        better_values=frozenset({"No"}),
        alternative_label="mailed billing",
    ),
    ActionableFeature(
        feature="tenure",
        kind="numeric",
        action="Provide onboarding help and an early loyalty benefit.",
        actionability=0.85,
        direction="newer",
        alternative_label="onboarding and loyalty support",
    ),
]

# Never recommend changing these (protected / personal / non-actionable).
NON_ACTIONABLE_FEATURES = frozenset(
    {"gender", "SeniorCitizen", "Partner", "Dependents", "customerID"}
)
