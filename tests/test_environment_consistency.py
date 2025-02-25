import unittest

from constants import SnakeConfig, create_default_environment


class EnvironmentConsistencyTest(unittest.TestCase):
    """Tests to verify environment consistency between training and play modes."""

    def setUp(self):
        """Set up environments for testing."""
        # Create environments with the same configuration
        self.train_env = create_default_environment(SnakeConfig.RENDER_MODE_NONE)
        self.play_env = create_default_environment(SnakeConfig.RENDER_MODE_NONE)

    def test_initial_snake_position_and_direction(self):
        """Verify initial snake position and direction are consistent."""
        # Reset both environments
        self.train_env.reset()
        self.play_env.reset()

        # Check snake positions
        self.assertEqual(
            self.train_env.snake,
            self.play_env.snake,
            "Initial snake positions differ between training and play environments",
        )

        # Check initial direction
        self.assertEqual(
            self.train_env.direction,
            self.play_env.direction,
            "Initial directions differ between training and play environments",
        )

    def test_episode_consistency(self):
        """Verify environment reset produces the same initial state across episodes."""
        # Get initial state from first reset
        self.train_env.reset()
        initial_snake = list(self.train_env.snake)  # Create a copy
        initial_direction = self.train_env.direction
        initial_food = self.train_env.food

        # Reset multiple times and check consistency
        for i in range(5):
            self.train_env.reset()
            self.assertEqual(
                initial_snake,
                self.train_env.snake,
                f"Snake position not consistent on reset {i+1}",
            )
            self.assertEqual(
                initial_direction,
                self.train_env.direction,
                f"Direction not consistent on reset {i+1}",
            )
            self.assertEqual(
                initial_food,
                self.train_env.food,
                f"Food position not consistent on reset {i+1}",
            )


if __name__ == "__main__":
    unittest.main()
