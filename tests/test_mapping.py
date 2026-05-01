from pm_rag import regex_mapping


def test_substring_match_case_insensitive() -> None:
    symbols = [
        "handlers.payment.payment_settled",
        "handlers.invoice.GENERATE_invoice",
        "utils.money.format_amount",
    ]
    m = regex_mapping(["payment_settled", "generate_invoice"], symbols)
    assert m["payment_settled"] == [0]
    assert m["generate_invoice"] == [1]


def test_no_match_yields_empty_list() -> None:
    m = regex_mapping(["nonexistent_event"], ["a", "b"])
    assert m["nonexistent_event"] == []


def test_duplicates_collapsed() -> None:
    m = regex_mapping(["x", "x", "x"], ["x_handler"])
    assert m == {"x": [0]}


def test_regex_chars_in_event_are_escaped() -> None:
    # `.` would normally be a wildcard
    m = regex_mapping(["a.b"], ["a.b", "axb"])
    assert m["a.b"] == [0]
