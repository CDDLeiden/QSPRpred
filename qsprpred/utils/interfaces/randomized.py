class Randomized:
    """An object with one or more pseudorandom actions that can be fixed with a seed.

    Attributes:
        seed (int | None):
            The seed to use to randomize the action. If `None`,
            a random seed is used instead of a fixed one (default: `None`).
    """
    def __init__(self, seed: int | None = None) -> None:
        """Create a new randomized action.

        Args:
            seed:
                the seed to use to randomize the action. If `None`,
                a random seed is used instead of a fixed one (default: `None`).
        """
        self.seed = seed

    def setSeed(self, seed: int | None = None):
        self.seed = seed

    def getSeed(self):
        """Get the seed used to randomize the action."""
        return self.seed
