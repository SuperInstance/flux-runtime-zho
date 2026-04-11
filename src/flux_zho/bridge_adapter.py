"""
流星桥接适配器 — FLUX-ZHO Bridge Adapter

Exposes the Chinese classifier type system (量词类型系统) to the
A2A type-safe cross-language bridge. Maps Chinese classifier-based
noun categorization to a universal type representation.

量词是中文最美丽的类型系统 — each classifier declares a noun's
type category, which we map to universal categories for inter-runtime
type-safe communication.

Interface:
    adapter = ZhoBridgeAdapter()
    types = adapter.export_types()           # list[UniversalType]
    local = adapter.import_type(universal)    # ZhoTypeSignature
    cost = adapter.bridge_cost("deu")         # BridgeCost
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from flux_zho.classifier_type import (
    ClassifierType,
    CLASSIFIER_TO_TYPE,
    CLASSIFIER_TYPE_NAMES,
    CLASSIFIER_NOUN_DB,
    NOUN_IMPLIED_TYPE,
)


# ══════════════════════════════════════════════════════════════════════
# Common bridge types (defined locally to avoid cross-repo dependencies)
# ══════════════════════════════════════════════════════════════════════

@dataclass
class BridgeCost:
    """Estimated cost of bridging to another runtime's type system.

    Attributes:
        numeric_cost: 0.0 = free/isomorphic, 1.0 = impossible
        information_loss: list of what gets lost in translation
        ambiguity_warnings: what becomes ambiguous
    """
    numeric_cost: float
    information_loss: list[str] = field(default_factory=list)
    ambiguity_warnings: list[str] = field(default_factory=list)


@dataclass
class UniversalType:
    """A paradigm-independent type representation for cross-language bridging.

    Attributes:
        paradigm: source paradigm name (e.g. "zho", "deu", "san")
        category: universal category (Agent, Patient, Temporal, etc.)
        constraints: paradigm-specific constraint details
        confidence: 0.0-1.0 mapping confidence
    """
    paradigm: str
    category: str
    constraints: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


class BridgeAdapter(ABC):
    """Abstract base for all bridge adapters across FLUX runtimes."""

    @abstractmethod
    def export_types(self) -> list[UniversalType]: ...

    @abstractmethod
    def import_type(self, universal: UniversalType) -> Any: ...

    @abstractmethod
    def bridge_cost(self, target_lang: str) -> BridgeCost: ...


# ══════════════════════════════════════════════════════════════════════
# ZhoTypeSignature — Chinese type representation for bridging
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ZhoTypeSignature:
    """Represents a Chinese classifier-based type for bridge export/import.

    In Chinese, every noun must be preceded by a classifier (量词) that
    declares its type category. This signature captures:
      - classifier: the specific Chinese classifier character (个, 只, 本, etc.)
      - classifier_type: the enum type it maps to
      - noun_category: human-readable category name
      - quantifier_scope: how the noun can be quantified

    Attributes:
        classifier: Chinese classifier character (e.g. 只, 本, 台)
        classifier_type: internal ClassifierType enum value
        noun_category: human-readable category name (e.g. "动物", "书籍")
        quantifier_scope: what quantification patterns this type supports
        confidence: how confident we are in this classification
    """
    classifier: str
    classifier_type: ClassifierType
    noun_category: str
    quantifier_scope: str = "countable"
    confidence: float = 1.0

    @property
    def type_name(self) -> str:
        return CLASSIFIER_TYPE_NAMES.get(self.classifier_type, "未知")


# ══════════════════════════════════════════════════════════════════════
# Classifier → Universal Type Mapping
# ══════════════════════════════════════════════════════════════════════

# Maps ClassifierType to universal categories
_CLASSIFIER_TO_UNIVERSAL: dict[ClassifierType, tuple[str, str, float]] = {
    ClassifierType.ANY:             ("Countable", "Generic untyped noun", 0.5),
    ClassifierType.ANIMAL:          ("Animate", "Living creature", 0.95),
    ClassifierType.PERSON:          ("Agent", "Human person (neutral)", 0.95),
    ClassifierType.PERSON_RESPECT:  ("Agent", "Human person (respectful)", 0.95),
    ClassifierType.BOOK:            ("BoundVolume", "Book, document, or text", 0.9),
    ClassifierType.MACHINE:         ("Artifact", "Machine or device", 0.9),
    ClassifierType.SET:             ("Collection", "Set or ensemble", 0.85),
    ClassifierType.VESSEL:          ("Vehicle", "Ship or watercraft", 0.95),
    ClassifierType.LONG:            ("Elongated", "Long/sequential object", 0.85),
    ClassifierType.FLAT:            ("Planar", "Flat surface or plane", 0.85),
    ClassifierType.PIECE:           ("Mass", "Block or mass noun", 0.8),
    ClassifierType.STICK:           ("Elongated", "Thin elongated object", 0.85),
    ClassifierType.ROUND:           ("Discrete", "Small round object", 0.8),
    ClassifierType.GRAIN:           ("Discrete", "Granular substance", 0.75),
    ClassifierType.DROP:            ("Continuous", "Liquid measure", 0.7),
    ClassifierType.COUNTER:         ("Event", "Count of actions/events", 0.9),
    ClassifierType.STEP:            ("Event", "Procedural step", 0.9),
    ClassifierType.KIND:            ("Abstract", "Category or kind", 0.85),
    ClassifierType.ITEM:            ("Artifact", "General item or affair", 0.7),
    ClassifierType.PLANE:           ("Planar", "Planar region", 0.8),
    ClassifierType.SPEED:           ("Measurement", "Rate of change", 0.9),
    ClassifierType.DISTANCE:        ("Measurement", "Distance measure", 0.9),
    ClassifierType.WEIGHT:          ("Measurement", "Weight measure", 0.9),
    ClassifierType.TIME:            ("Temporal", "Time duration", 0.95),
    ClassifierType.MESSAGE:         ("Information", "Message or communication", 0.9),
    ClassifierType.REPORT:          ("Information", "Report or document", 0.85),
    ClassifierType.CLAUSE:          ("Structural", "Clause or contractual item", 0.85),
    ClassifierType.CONTAINER:       ("Artifact", "Container or vessel", 0.8),
    ClassifierType.AIRCRAFT:        ("Vehicle", "Aircraft or frame", 0.95),
    ClassifierType.ANCHOR:          ("Artifact", "Maritime anchor", 0.95),
    ClassifierType.SAIL:            ("Artifact", "Sail or sail-like", 0.95),
    ClassifierType.ROUND_TRIP:      ("Event", "Round or cycle", 0.85),
    ClassifierType.FRAMEWORK:       ("Structural", "Framework or skeleton", 0.85),
}

# Reverse map: universal category → best-matching ClassifierType
_UNIVERSAL_TO_CLASSIFIER: dict[str, ClassifierType] = {
    "Countable":   ClassifierType.ANY,
    "Animate":     ClassifierType.ANIMAL,
    "Agent":       ClassifierType.PERSON_RESPECT,
    "BoundVolume": ClassifierType.BOOK,
    "Artifact":    ClassifierType.MACHINE,
    "Collection":  ClassifierType.SET,
    "Vehicle":     ClassifierType.VESSEL,
    "Elongated":   ClassifierType.LONG,
    "Planar":      ClassifierType.FLAT,
    "Mass":        ClassifierType.PIECE,
    "Discrete":    ClassifierType.ROUND,
    "Continuous":  ClassifierType.DROP,
    "Event":       ClassifierType.COUNTER,
    "Abstract":    ClassifierType.KIND,
    "Temporal":    ClassifierType.TIME,
    "Information": ClassifierType.MESSAGE,
    "Structural":  ClassifierType.CLAUSE,
    "Measurement": ClassifierType.SPEED,
}

# Quantifier scope mapping
_CLASSIFIER_TO_QUANTIFIER_SCOPE: dict[ClassifierType, str] = {
    ClassifierType.COUNTER:      "event_count",
    ClassifierType.STEP:         "procedure",
    ClassifierType.SPEED:        "rate",
    ClassifierType.DISTANCE:     "linear_measure",
    ClassifierType.WEIGHT:       "mass_measure",
    ClassifierType.TIME:         "temporal_measure",
    ClassifierType.DROP:         "continuous",
    ClassifierType.GRAIN:        "granular",
    ClassifierType.ROUND_TRIP:   "cyclic",
}


# ══════════════════════════════════════════════════════════════════════
# Language affinity estimates for bridge_cost
# ══════════════════════════════════════════════════════════════════════

_LANG_AFFINITY: dict[str, dict[str, Any]] = {
    "zho": {"cost": 0.0, "loss": [], "ambiguity": []},
    "wen": {"cost": 0.1, "loss": [], "ambiguity": ["Simplified vs Classical distinction lost"]},
    "deu": {"cost": 0.45, "loss": ["Classifier granularity", "Shape-based categorization"],
            "ambiguity": ["German gender has no direct classifier equivalent"]},
    "kor": {"cost": 0.40, "loss": ["Classifier particle system", "Noun shape category"],
            "ambiguity": ["Korean counters less specific than Chinese classifiers"]},
    "san": {"cost": 0.55, "loss": ["All classifier distinctions", "Noun shape semantics"],
            "ambiguity": ["Sanskrit has no classifier system — nouns map via vibhakti only"]},
    "lat": {"cost": 0.50, "loss": ["Classifier granularity", "Shape-based typing"],
            "ambiguity": ["Latin case/gender system is structurally different"]},
}


# ══════════════════════════════════════════════════════════════════════
# ZhoBridgeAdapter — Main adapter class
# ══════════════════════════════════════════════════════════════════════

class ZhoBridgeAdapter(BridgeAdapter):
    """Bridge adapter for the Chinese (中文) classifier type system.

    Exports all registered classifier types as UniversalType instances
    and can import universal types back into Chinese classifier categories.

    Usage:
        adapter = ZhoBridgeAdapter()
        types = adapter.export_types()
        cost = adapter.bridge_cost("deu")
    """

    PARADIGM = "zho"

    def export_types(self) -> list[UniversalType]:
        """Export all registered classifier types as universal types.

        Returns:
            List of UniversalType for each unique ClassifierType value
            found in the CLASSIFIER_TO_TYPE mapping.
        """
        exported: list[UniversalType] = []
        seen: set[ClassifierType] = set()

        for clf_char, clf_type in CLASSIFIER_TO_TYPE.items():
            if clf_type in seen:
                continue
            seen.add(clf_type)

            cat, desc, conf = _CLASSIFIER_TO_UNIVERSAL.get(
                clf_type, ("Unknown", "Unclassified", 0.3)
            )

            # Collect sample nouns for this classifier
            sample_nouns = CLASSIFIER_NOUN_DB.get(clf_char, [])[:5]

            exported.append(UniversalType(
                paradigm=self.PARADIGM,
                category=cat,
                constraints={
                    "classifier": clf_char,
                    "classifier_type": clf_type.name,
                    "noun_category": CLASSIFIER_TYPE_NAMES.get(clf_type, ""),
                    "quantifier_scope": _CLASSIFIER_TO_QUANTIFIER_SCOPE.get(
                        clf_type, "countable"
                    ),
                    "description": desc,
                    "sample_nouns": sample_nouns,
                },
                confidence=conf,
            ))

        return exported

    def import_type(self, universal: UniversalType) -> ZhoTypeSignature:
        """Import a universal type into the Chinese classifier system.

        Maps a UniversalType's category back to the best-matching
        Chinese classifier and produces a ZhoTypeSignature.

        Args:
            universal: A UniversalType from another runtime

        Returns:
            ZhoTypeSignature with the best-matching classifier
        """
        category = universal.category

        # Direct match
        clf_type = _UNIVERSAL_TO_CLASSIFIER.get(category)

        # Fallback: try partial match on category name
        if clf_type is None:
            for uni_cat, ct in _UNIVERSAL_TO_CLASSIFIER.items():
                if uni_cat.lower() in category.lower():
                    clf_type = ct
                    break

        # Ultimate fallback
        if clf_type is None:
            clf_type = ClassifierType.ANY

        # Find a representative classifier character for this type
        clf_char = "个"  # default generic
        for char, ct in CLASSIFIER_TO_TYPE.items():
            if ct == clf_type and char != "个":
                clf_char = char
                break

        # Apply confidence from the source
        confidence = universal.confidence * 0.9  # slight loss on import

        # Determine quantifier scope
        scope = _CLASSIFIER_TO_QUANTIFIER_SCOPE.get(clf_type, "countable")

        return ZhoTypeSignature(
            classifier=clf_char,
            classifier_type=clf_type,
            noun_category=CLASSIFIER_TYPE_NAMES.get(clf_type, ""),
            quantifier_scope=scope,
            confidence=confidence,
        )

    def bridge_cost(self, target_lang: str) -> BridgeCost:
        """Estimate the cost of bridging to another runtime.

        Args:
            target_lang: Target language code (e.g. "deu", "san", "lat")

        Returns:
            BridgeCost with estimated numeric cost, information loss,
            and ambiguity warnings
        """
        target = target_lang.lower().strip()

        if target == self.PARADIGM:
            return BridgeCost(
                numeric_cost=0.0,
                information_loss=[],
                ambiguity_warnings=[],
            )

        affinity = _LANG_AFFINITY.get(target, {
            "cost": 0.6,
            "loss": ["All classifier distinctions"],
            "ambiguity": ["Unknown target language — assume high loss"],
        })

        return BridgeCost(
            numeric_cost=affinity["cost"],
            information_loss=list(affinity["loss"]),
            ambiguity_warnings=list(affinity["ambiguity"]),
        )

    def resolve_classifier(self, text: str) -> ZhoTypeSignature:
        """Resolve a Chinese text fragment to a ZhoTypeSignature.

        Convenience method that parses "数字+量词+名词" patterns.

        Args:
            text: Chinese text (e.g. "三只猫", "一本書")

        Returns:
            ZhoTypeSignature extracted from the text
        """
        from flux_zho.classifier_type import ClassifierTypeSolver

        solver = ClassifierTypeSolver()
        resolution = solver.resolve(text)

        scope = _CLASSIFIER_TO_QUANTIFIER_SCOPE.get(
            resolution.classifier_type, "countable"
        )

        return ZhoTypeSignature(
            classifier=resolution.classifier,
            classifier_type=resolution.classifier_type,
            noun_category=resolution.type_name,
            quantifier_scope=scope,
            confidence=resolution.confidence,
        )
