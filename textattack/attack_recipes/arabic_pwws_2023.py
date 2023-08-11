"""

PWWS Arabic version 2023
=======

(Generating Natural Language Adversarial Examples through Probability Weighted Word Saliency)

"""
from textattack import Attack
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import WordSwapWordNet

from .attack_recipe import AttackRecipe


class ArabicPWWS2023(AttackRecipe):
    """An implementation of Probability Weighted Word Saliency from "Generating
    Natural Language Adversarial Examples through Probability Weighted Word
    Saliency", Ren et al., 2019.

    Arabic version 2023

    """

    @staticmethod
    def build(model_wrapper, stopwords=None):
        transformation = WordSwapWordNet()

        constraints = [RepeatModification()]
        if stopwords is not None:
            constraints.append(StopwordModification(stopwords=stopwords))
        else:
            constraints.append(StopwordModification(language="arabic"))

        goal_function = UntargetedClassification(model_wrapper)
        # search over words based on a combination of their saliency score, and how efficient the WordSwap transform is
        search_method = GreedyWordSwapWIR("weighted-saliency")
        return Attack(goal_function, constraints, transformation, search_method)

