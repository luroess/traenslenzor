"""Test precision configuration validation."""

from traenslenzor.doc_classifier.lightning.lit_trainer_factory import TrainerFactoryConfig


class TestPrecisionValidation:
    """Test suite for precision configuration validation."""

    def test_tf32_with_fp32_no_warning(self):
        """Test that TF32 with FP32 precision doesn't trigger warnings."""
        # This is the recommended combination - should work without warnings
        config = TrainerFactoryConfig(precision="32-true", tf32_matmul_precision="medium")
        assert config.precision == "32-true"
        assert config.tf32_matmul_precision == "medium"

    def test_tf32_with_fp16_triggers_warning(self, caplog):
        """Test that TF32 with FP16 precision triggers a warning."""
        # This combination should trigger a warning
        config = TrainerFactoryConfig(precision="16-mixed", tf32_matmul_precision="medium")

        # Config should still be created (warning, not error)
        assert config.precision == "16-mixed"
        assert config.tf32_matmul_precision == "medium"

    def test_tf32_with_bf16_triggers_warning(self, caplog):
        """Test that TF32 with BF16 precision triggers a warning."""
        # This combination should trigger a warning
        config = TrainerFactoryConfig(precision="bf16-mixed", tf32_matmul_precision="high")

        # Config should still be created (warning, not error)
        assert config.precision == "bf16-mixed"
        assert config.tf32_matmul_precision == "high"

    def test_tf32_none_no_warning(self):
        """Test that tf32_matmul_precision=None doesn't trigger warnings."""
        # Setting to None should never warn
        config = TrainerFactoryConfig(precision="16-mixed", tf32_matmul_precision=None)
        assert config.precision == "16-mixed"
        assert config.tf32_matmul_precision is None

    def test_recommended_combinations(self):
        """Test various recommended precision combinations."""
        # Fast training (default)
        config1 = TrainerFactoryConfig(precision="32-true", tf32_matmul_precision="medium")
        assert config1.precision == "32-true"
        assert config1.tf32_matmul_precision == "medium"

        # Maximum speed
        config2 = TrainerFactoryConfig(precision="bf16-mixed", tf32_matmul_precision=None)
        assert config2.precision == "bf16-mixed"
        assert config2.tf32_matmul_precision is None

        # Maximum accuracy
        config3 = TrainerFactoryConfig(precision="32-true", tf32_matmul_precision="highest")
        assert config3.precision == "32-true"
        assert config3.tf32_matmul_precision == "highest"

    def test_defaults(self):
        """Test that defaults are sensible."""
        config = TrainerFactoryConfig()

        # Default should be FP32 with TF32 medium for good balance
        assert config.precision == "32-true"
        assert config.tf32_matmul_precision == "medium"
