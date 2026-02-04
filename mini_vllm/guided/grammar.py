from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Union
import json
import re

@dataclass
class GrammarRule:

    name: str
    pattern: str
    is_terminal: bool = True
    alternatives: Optional[List[str]] = None

class JSONSchemaParser:

    STRING_PATTERN = r'"([^"\\]|\\.)*"'
    INTEGER_PATTERN = r'-?(?:0|[1-9]\d*)'
    NUMBER_PATTERN = r'-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?'
    BOOLEAN_PATTERN = r'(?:true|false)'
    NULL_PATTERN = r'null'
    WHITESPACE_PATTERN = r'[ \t\n\r]*'

    def __init__(self, allow_whitespace: bool = True):
        self.allow_whitespace = allow_whitespace
        self._rules: Dict[str, GrammarRule] = {}
        self._rule_counter = 0

    def parse(self, schema: Dict[str, Any]) -> "Grammar":
        self._rules = {}
        self._rule_counter = 0

        root_rule = self._schema_to_rule(schema, "root")

        return Grammar(
            rules=self._rules,
            root_rule=root_rule,
            schema=schema,
        )

    def _schema_to_rule(
        self,
        schema: Dict[str, Any],
        name: str,
    ) -> str:
        schema_type = schema.get("type")

        if "enum" in schema:
            return self._enum_rule(schema["enum"], name)

        if "const" in schema:
            return self._const_rule(schema["const"], name)

        if "anyOf" in schema:
            return self._anyof_rule(schema["anyOf"], name)
        if "oneOf" in schema:
            return self._anyof_rule(schema["oneOf"], name)

        if schema_type == "string":
            return self._string_rule(schema, name)
        elif schema_type == "integer":
            return self._integer_rule(schema, name)
        elif schema_type == "number":
            return self._number_rule(schema, name)
        elif schema_type == "boolean":
            return self._boolean_rule(name)
        elif schema_type == "null":
            return self._null_rule(name)
        elif schema_type == "object":
            return self._object_rule(schema, name)
        elif schema_type == "array":
            return self._array_rule(schema, name)
        elif isinstance(schema_type, list):

            return self._multi_type_rule(schema, name)
        else:

            return self._any_json_rule(name)

    def _string_rule(self, schema: Dict, name: str) -> str:

        if "pattern" in schema:
            pattern = f'"{schema["pattern"]}"'
        else:
            pattern = self.STRING_PATTERN

        rule = GrammarRule(name=name, pattern=pattern)
        self._rules[name] = rule
        return name

    def _integer_rule(self, schema: Dict, name: str) -> str:
        rule = GrammarRule(name=name, pattern=self.INTEGER_PATTERN)
        self._rules[name] = rule
        return name

    def _number_rule(self, schema: Dict, name: str) -> str:
        rule = GrammarRule(name=name, pattern=self.NUMBER_PATTERN)
        self._rules[name] = rule
        return name

    def _boolean_rule(self, name: str) -> str:
        rule = GrammarRule(name=name, pattern=self.BOOLEAN_PATTERN)
        self._rules[name] = rule
        return name

    def _null_rule(self, name: str) -> str:
        rule = GrammarRule(name=name, pattern=self.NULL_PATTERN)
        self._rules[name] = rule
        return name

    def _enum_rule(self, values: List, name: str) -> str:

        patterns = [json.dumps(v) for v in values]
        pattern = f"({'|'.join(re.escape(p) for p in patterns)})"

        rule = GrammarRule(
            name=name,
            pattern=pattern,
            alternatives=patterns,
        )
        self._rules[name] = rule
        return name

    def _const_rule(self, value: Any, name: str) -> str:
        pattern = re.escape(json.dumps(value))
        rule = GrammarRule(name=name, pattern=pattern)
        self._rules[name] = rule
        return name

    def _object_rule(self, schema: Dict, name: str) -> str:
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        additional = schema.get("additionalProperties", True)

        if not properties:

            pattern = r'\{' + self.WHITESPACE_PATTERN + r'\}'
            rule = GrammarRule(name=name, pattern=pattern, is_terminal=False)
            self._rules[name] = rule
            return name

        prop_patterns = []
        for prop_name, prop_schema in properties.items():
            prop_rule_name = f"{name}_prop_{prop_name}"
            prop_rule = self._schema_to_rule(prop_schema, prop_rule_name)

            ws = self.WHITESPACE_PATTERN if self.allow_whitespace else ""
            prop_pattern = f'"{prop_name}"{ws}:{ws}{{{prop_rule}}}'

            if prop_name in required:
                prop_patterns.append(prop_pattern)
            else:
                prop_patterns.append(f"({prop_pattern})?")

        ws = self.WHITESPACE_PATTERN if self.allow_whitespace else ""
        separator = f"{ws},{ws}"
        content = separator.join(prop_patterns)
        pattern = r'\{' + ws + content + ws + r'\}'

        rule = GrammarRule(name=name, pattern=pattern, is_terminal=False)
        self._rules[name] = rule
        return name

    def _array_rule(self, schema: Dict, name: str) -> str:
        items_schema = schema.get("items", {})
        items_rule_name = f"{name}_item"
        items_rule = self._schema_to_rule(items_schema, items_rule_name)

        ws = self.WHITESPACE_PATTERN if self.allow_whitespace else ""
        separator = f"{ws},{ws}"

        pattern = (
            r'\[' + ws +
            f"({{{items_rule}}}({separator}{{{items_rule}}})*)?{ws}" +
            r'\]'
        )

        rule = GrammarRule(name=name, pattern=pattern, is_terminal=False)
        self._rules[name] = rule
        return name

    def _anyof_rule(self, schemas: List[Dict], name: str) -> str:
        sub_rules = []
        for i, sub_schema in enumerate(schemas):
            sub_name = f"{name}_alt{i}"
            sub_rule = self._schema_to_rule(sub_schema, sub_name)
            sub_rules.append(sub_rule)

        pattern = f"({'|'.join(f'{{{r}}}' for r in sub_rules)})"
        rule = GrammarRule(
            name=name,
            pattern=pattern,
            is_terminal=False,
            alternatives=sub_rules,
        )
        self._rules[name] = rule
        return name

    def _multi_type_rule(self, schema: Dict, name: str) -> str:
        types = schema["type"]
        sub_rules = []

        for i, t in enumerate(types):
            sub_schema = {**schema, "type": t}
            sub_name = f"{name}_type{i}"
            sub_rule = self._schema_to_rule(sub_schema, sub_name)
            sub_rules.append(sub_rule)

        pattern = f"({'|'.join(f'{{{r}}}' for r in sub_rules)})"
        rule = GrammarRule(name=name, pattern=pattern, alternatives=sub_rules)
        self._rules[name] = rule
        return name

    def _any_json_rule(self, name: str) -> str:

        pattern = (
            f"({self.STRING_PATTERN}|{self.NUMBER_PATTERN}|"
            f"{self.BOOLEAN_PATTERN}|{self.NULL_PATTERN}|"
            r'\{[^}]*\}|\[[^\]]*\])'
        )
        rule = GrammarRule(name=name, pattern=pattern)
        self._rules[name] = rule
        return name

@dataclass
class Grammar:

    rules: Dict[str, GrammarRule]
    root_rule: str
    schema: Dict[str, Any]

    def get_rule(self, name: str) -> Optional[GrammarRule]:
        return self.rules.get(name)

    def get_root_pattern(self) -> str:
        return self.rules[self.root_rule].pattern

    def validate_json(self, text: str) -> bool:
        try:
            data = json.loads(text)

            return self._validate_against_schema(data, self.schema)
        except json.JSONDecodeError:
            return False

    def _validate_against_schema(
        self,
        data: Any,
        schema: Dict[str, Any],
    ) -> bool:
        schema_type = schema.get("type")

        if "enum" in schema:
            return data in schema["enum"]

        if "const" in schema:
            return data == schema["const"]

        if schema_type == "string":
            return isinstance(data, str)
        elif schema_type == "integer":
            return isinstance(data, int) and not isinstance(data, bool)
        elif schema_type == "number":
            return isinstance(data, (int, float)) and not isinstance(data, bool)
        elif schema_type == "boolean":
            return isinstance(data, bool)
        elif schema_type == "null":
            return data is None
        elif schema_type == "object":
            if not isinstance(data, dict):
                return False

            required = schema.get("required", [])
            for prop in required:
                if prop not in data:
                    return False

            properties = schema.get("properties", {})
            for prop, value in data.items():
                if prop in properties:
                    if not self._validate_against_schema(value, properties[prop]):
                        return False
            return True
        elif schema_type == "array":
            if not isinstance(data, list):
                return False
            items_schema = schema.get("items", {})
            for item in data:
                if not self._validate_against_schema(item, items_schema):
                    return False
            return True

        return True

def parse_json_schema(schema: Union[str, Dict]) -> Grammar:
    if isinstance(schema, str):
        schema = json.loads(schema)

    parser = JSONSchemaParser()
    return parser.parse(schema)
