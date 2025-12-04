from __future__ import annotations

from calt.data_loader.utils.preprocessor import AbstractPreprocessor


class PlainTextPreprocessor(AbstractPreprocessor):
    """
    Convert whitespace-separated integer sequences into CALT's internal token form.

    CALT のトークナイザは ``C<num>`` という記法を前提に語彙を構成しているため、
    数値列データをそのまま与えると語彙外トークンになってしまう。
    このプリプロセッサは入力文字列中の整数を ``C<num>`` へ変換し、逆変換も提供する。
    """

    def __init__(self, max_coeff: int) -> None:
        super().__init__(num_variables=0, max_degree=0, max_coeff=max_coeff)

    def encode(self, text: str) -> str:
        stripped = text.strip()
        if not stripped:
            return "C0"
        tokens = []
        for chunk in stripped.split():
            if not chunk:
                continue
            try:
                int(chunk)
            except ValueError as exc:
                raise ValueError(f"Non-integer token '{chunk}' encountered.") from exc
            tokens.append(f"C{chunk}")
        return " ".join(tokens) if tokens else "C0"

    def decode(self, tokens: str) -> str:
        stripped = tokens.strip()
        if not stripped:
            return "0"
        numbers = []
        for token in stripped.split():
            if not token.startswith("C"):
                raise ValueError(f"Unexpected token '{token}'.")
            numbers.append(token[1:])
        return " ".join(numbers) if numbers else "0"

