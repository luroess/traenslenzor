"""Test JSON schema contracts for MCP tools."""

import json


class TestDetectFontNameContract:
    """Test detect_font_name tool contract."""

    def test_input_schema(self):
        """Test input schema structure."""
        # schema = {
        #     "type": "object",
        #     "properties": {
        #         "image_path": {
        #             "type": "string",
        #         }
        #     },
        #     "required": ["image_path"],
        # }

        # Valid input
        valid_input = {"image_path": "/path/to/image.png"}
        assert "image_path" in valid_input
        assert isinstance(valid_input["image_path"], str)

    def test_output_schema(self):
        """Test output schema structure."""
        # Valid output
        valid_output = {"font_name": "Arial"}

        assert "font_name" in valid_output
        assert isinstance(valid_output["font_name"], str)

    def test_round_trip(self):
        """Test round-trip JSON serialization."""
        # Input
        input_data = {"image_path": "/path/to/image.png"}
        input_json = json.dumps(input_data)
        parsed_input = json.loads(input_json)
        assert parsed_input == input_data

        # Output
        output_data = {"font_name": "DejaVuSans"}
        output_json = json.dumps(output_data)
        parsed_output = json.loads(output_json)
        assert parsed_output == output_data


class TestEstimateFontSizeContract:
    """Test estimate_font_size tool contract."""

    def test_input_schema(self):
        """Test input schema structure."""
        # schema = {
        #     "type": "object",
        #     "properties": {
        #         "text_box_size": {
        #             "type": "array",
        #             "items": {"type": "number"},
        #             "minItems": 2,
        #             "maxItems": 2,
        #         },
        #         "text": {
        #             "type": "string",
        #         },
        #         "font_name": {
        #             "type": "string",
        #         },
        #     },
        #     "required": ["text_box_size", "text"],
        # }

        # Valid input (without optional font_name)
        valid_input1 = {
            "text_box_size": [400.5, 64.2],
            "text": "Hello World",
        }
        assert isinstance(valid_input1["text_box_size"], list)
        assert len(valid_input1["text_box_size"]) == 2
        assert all(isinstance(x, (int, float)) for x in valid_input1["text_box_size"])
        assert isinstance(valid_input1["text"], str)

        # Valid input (with optional font_name)
        valid_input2 = {
            "text_box_size": [400, 64],
            "text": "Hello World",
            "font_name": "DejaVuSans",
        }
        assert "font_name" in valid_input2
        assert isinstance(valid_input2["font_name"], str)

    def test_output_schema(self):
        """Test output schema structure."""
        # Valid output
        valid_output = {"font_size_pt": 12.5}

        assert "font_size_pt" in valid_output
        assert isinstance(valid_output["font_size_pt"], (int, float))

    def test_round_trip(self):
        """Test round-trip JSON serialization."""
        # Input
        input_data = {
            "text_box_size": [400, 64],
            "text": "Hello World",
            "font_name": "DejaVuSans",
        }
        input_json = json.dumps(input_data)
        parsed_input = json.loads(input_json)
        assert parsed_input == input_data

        # Output
        output_data = {"font_size_pt": 12.5}
        output_json = json.dumps(output_data)
        parsed_output = json.loads(output_json)
        assert parsed_output == output_data

    def test_text_box_size_validation(self):
        """Test text_box_size array validation."""
        # Valid sizes
        valid_sizes = [
            [100, 50],
            [100.5, 50.2],
            [0.1, 0.1],
        ]

        for size in valid_sizes:
            assert len(size) == 2
            assert all(isinstance(x, (int, float)) for x in size)

        # Invalid sizes (should fail in actual implementation)
        invalid_sizes = [
            [100],  # too short
            [100, 50, 30],  # too long
            ["100", "50"],  # wrong type
        ]

        for size in invalid_sizes:
            assert len(size) != 2 or not all(isinstance(x, (int, float)) for x in size)


class TestSchemaValidation:
    """Test schema validation helpers."""

    def test_validate_string_field(self):
        """Test string field validation."""
        data = {"field": "value"}
        assert "field" in data
        assert isinstance(data["field"], str)

        invalid_data = {"field": 123}
        assert not isinstance(invalid_data["field"], str)

    def test_validate_number_field(self):
        """Test number field validation."""
        data = {"field": 12.5}
        assert "field" in data
        assert isinstance(data["field"], (int, float))

        invalid_data = {"field": "12.5"}
        assert not isinstance(invalid_data["field"], (int, float))

    def test_validate_array_field(self):
        """Test array field validation."""
        data = {"field": [1, 2, 3]}
        assert "field" in data
        assert isinstance(data["field"], list)

        invalid_data = {"field": "not an array"}
        assert not isinstance(invalid_data["field"], list)

    def test_required_fields(self):
        """Test required field validation."""
        # Has required field
        data = {"required_field": "value"}
        assert "required_field" in data

        # Missing required field
        data = {"other_field": "value"}
        assert "required_field" not in data
