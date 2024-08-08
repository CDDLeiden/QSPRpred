from abc import abstractmethod


class Randomized:
    """An object with one or more pseudorandom actions that can be fixed with a seed.
    """

    @property
    @abstractmethod
    def randomState(self) -> int:
        """Get the random state for the object."""

    @randomState.setter
    @abstractmethod
    def randomState(self, seed: int | None):
        """Set the random state for the object.

        Args:
            seed (int | None):
                The seed to use to randomize the action. If `None`,
                a random seed is used instead of a fixed one.
        """
