from typing import List

class Triple:
    """
    Represents an atom in a knowledge graph consisting of a subject, predicate, and object.
    """

    def __init__(self, subject: str, predicate: str, object: str):
        """
        Initializes a Triple instance with possible variables and possibly with prefixes.
        
        Parameters:
        - subject (str): The ID of the subject, could also be a variable.
        - predicate (str): The ID of the predicate.
        - object (str): The ID of the object, could also be a variable.
        """

        self.subject = subject
        self.predicate = predicate
        self.object = object

    def get_subject(self):
        return self.subject
    
    def set_subject(self, subject):
        self.subject = subject

    def get_predicate(self):
        return self.predicate

    def get_object(self):
        return self.object
    
    def set_object(self, object):
        self.object = object

    def __str__(self) -> str:
        """Return a string representation of the Triple."""

        return f"Triple(subject={self.subject}, predicate={self.predicate}, object={self.object})"
    
    def __eq__(self, other) -> bool:
        """Check equality based on subject, predicate, and object."""
        if not isinstance(other, Triple):
            return False
        return (
            self.subject == other.subject and
            self.predicate == other.predicate and
            self.object == other.object
        )

    def __hash__(self) -> int:
        """Generate a hash based on subject, predicate, and object."""
        return hash((self.subject, self.predicate, self.object))


class Quintuple(Triple):
    """
    Represents an atom in a temporal knowledge graph consisting of a subject, predicate, object, start date, and end date.
    """

    def __init__(self, subject: str, predicate: str, object: str, start_date: str, end_date: str):
        """
        Initializes a Quintuple instance.
        
        Parameters:
        - subject (str): The ID of the subject, could also be a variable.
        - predicate (str): The ID of the predicate.
        - object (str): The ID of the object, could also be a variable.
        - start_date (str): The starting date (in YYYY-MM-DD format).
        - end_date (str): The ending date (in YYYY-MM-DD format).
        """

        super().__init__(subject, predicate, object)
        self.start_date = start_date
        self.end_date = end_date

    def get_triple(self):
        return Triple(self.subject, self.predicate, self.object)

    def get_interval(self):
        return self.start_date, self.end_date

    def __str__(self) -> str:
        """Return a string representation of the Quintuple."""

        return (f"Quintuple(subject={self.subject}, "
                f"predicate={self.predicate}, "
                f"object={self.object}, "
                f"start_date={self.start_date}, "
                f"end_date={self.end_date})")
    
    def __hash__(self) -> int:
        """Include start_date and end_date in the hash."""
        return hash((self.subject, self.predicate, self.object, self.start_date, self.end_date))

    def __eq__(self, other) -> bool:
        """Include start_date and end_date in equality check."""
        if not isinstance(other, Quintuple):
            return False
        return (
            self.subject == other.subject and
            self.predicate == other.predicate and
            self.object == other.object and
            self.start_date == other.start_date and
            self.end_date == other.end_date
        )


class Rule:
    def __init__(self, head: Triple, body: List[Triple], body_coverage: int, support: int, confidence: float):
        self.head = head
        self.body = body
        self.body_coverage = body_coverage
        self.support = support
        self.confidence = confidence

    def get_head(self) -> Triple:
        return self.head

    def get_body(self) -> List[Triple]:
        return self.body
    
    def get_body_coverage(self) -> int:
        return self.body_coverage
    
    def get_support(self) -> int:
        return self.support

    def get_confidence(self) -> float:
        return self.confidence

    def __str__(self) -> str:
        """Return a string representation of a Rule."""

        return (f"Rule(head={self.head}, "
                f"body={[str(b) for b in self.body]}, "
                f"body coverage={self.body_coverage}, "
                f"body and head coverage={self.support}, "
                f"confidence={self.confidence})")


def calculate_confidence_of_list_of_rules(rules: List[Rule], epsilon = 0.5):
    confidence = 0
    for level, rule in enumerate(rules, start=1):
        updated_confidence = confidence + (epsilon ** (level - 1)) * rule.get_confidence()
        confidence = updated_confidence

    return confidence


def get_confidence(rules: List[Rule]) -> float:
    return rules[0].get_confidence()


def substitute_variable_in_rule(rule: Rule, substitution: str, variable: str) -> Rule:
    # helper function
    def substitute_variable_in_triple(triple: Triple, substitution: str, variable: str) -> Triple:
        new_subject = substitution if triple.get_subject() == variable else triple.get_subject()
        new_object = substitution if triple.get_object() == variable else triple.get_object()

        return Triple(new_subject, triple.get_predicate(), new_object)
    
    head = substitute_variable_in_triple(rule.get_head(), substitution, variable)
    body_atoms = [substitute_variable_in_triple(body, substitution, variable) for body in rule.get_body()]

    return Rule(head, body_atoms, rule.get_body_coverage(), rule.get_support(), rule.get_confidence())


def rule_has_variable(rule: Rule) -> bool:
    if atom_has_variable(rule.get_head()):
        return True
    else:
        for body_atom in rule.get_body():
            if atom_has_variable(body_atom):
                return True
    return False


def atom_has_variable(atom: Triple) -> bool:
    return (
        atom.get_subject().isalpha() or 
        atom.get_object().isalpha()
    )


def retrieve_all_variables_rule(rule: Rule) -> List[str]:
    variables = set()

    # check head of the rule
    head = rule.get_head()
    if atom_has_variable(head):
        if head.get_subject().isalpha():
            variables.add(head.get_subject())
        if head.get_object().isalpha():
            variables.add(head.get_object())

    # check body of the rule
    for body_atom in rule.get_body():
        if atom_has_variable(body_atom):
            if body_atom.get_subject().isalpha():
                variables.add(body_atom.get_subject())
            if body_atom.get_object().isalpha():
                variables.add(body_atom.get_object())

    return list(variables)
