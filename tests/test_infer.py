from preference_consistency.infer import model_is_present


def test_model_is_present() -> None:
    assert model_is_present(["llama3.2:3b"], "llama3.2:3b")
    assert not model_is_present(["mistral:7b"], "llama3.2:3b")
