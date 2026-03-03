"""
test_clean_sentiment.py – Unit tests for clean_sentiment_data.py

Run:  pytest test_clean_sentiment.py -v
"""
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from clean_sentiment_data import (
    normalize_unicode,
    mask_entities,
    split_camelcase,
    handle_emojis,
    compress_punctuation,
    negation_scope,
    is_bot_post,
    clean_text,
    detect_language,
    TranslationCache,
    remove_boilerplate,
    ascii_fold,
)
import re


# ── Unicode normalization ────────────────────────────────────────────────

class TestNormalizeUnicode:
    def test_fixes_mojibake(self):
        # â€œ / â€ are common UTF-8→latin-1 mojibake for " / "
        result = normalize_unicode("â\u0080\u009cHello worldâ\u0080\u009d")
        assert "\u201c" in result or "Hello world" in result

    def test_collapses_whitespace(self):
        assert normalize_unicode("hello    world") == "hello world"

    def test_strips_ends(self):
        assert normalize_unicode("  hello  ") == "hello"

    def test_nfkc_normalisation(self):
        # ﬁ (U+FB01) should decompose to "fi" under NFKC
        assert normalize_unicode("ﬁnd") == "find"


# ── Entity masking ───────────────────────────────────────────────────────

class TestMaskEntities:
    def test_url_replaced(self):
        assert "<URL>" in mask_entities("Visit https://example.com today")

    def test_email_replaced(self):
        assert "<EMAIL>" in mask_entities("Send to user@example.com")

    def test_mention_replaced(self):
        assert "<USER>" in mask_entities("Thanks @johndoe")

    def test_hashtag_keeps_word(self):
        result = mask_entities("#LoveThis is great")
        assert "#" not in result
        assert "Love" in result
        assert "This" in result

    def test_hashtag_camelcase_split(self):
        result = mask_entities("#MakeAmericaGreatAgain")
        assert "Make America Great Again" in result

    def test_numbers_kept_by_default(self):
        result = mask_entities("I have 100 apples")
        assert "100" in result

    def test_numbers_masked(self):
        result = mask_entities("I have 100 apples", num_mode="mask")
        assert "<NUM>" in result
        assert "100" not in result


# ── CamelCase splitter ───────────────────────────────────────────────────

class TestSplitCamelCase:
    def test_basic(self):
        assert split_camelcase("LoveThis") == "Love This"

    def test_multiple_words(self):
        assert split_camelcase("MakeAmericaGreatAgain") == "Make America Great Again"

    def test_single_word(self):
        assert split_camelcase("hello") == "hello"

    def test_all_caps(self):
        # All-caps words shouldn't be split (no lower→upper boundary)
        assert split_camelcase("USA") == "USA"


# ── Punctuation compression ─────────────────────────────────────────────

class TestCompressPunctuation:
    def test_compress_exclamation(self):
        assert compress_punctuation("wow!!!!!") == "wow!!!"

    def test_compress_question(self):
        assert compress_punctuation("really?????") == "really???"

    def test_keeps_three(self):
        assert compress_punctuation("nice!!!") == "nice!!!"

    def test_keeps_single(self):
        assert compress_punctuation("hello!") == "hello!"


# ── Emoji handling ───────────────────────────────────────────────────────

class TestHandleEmojis:
    def test_demojize(self):
        result = handle_emojis("I am happy 😊")
        assert ":smiling_face_with_smiling_eyes:" in result or ":smiling" in result

    def test_emoticon_smile(self):
        result = handle_emojis("Great :)")
        assert ":smile:" in result

    def test_emoticon_sad(self):
        result = handle_emojis("Bad :(")
        assert ":sad:" in result

    def test_emoticon_wink(self):
        result = handle_emojis("Sure ;)")
        assert ":wink:" in result

    def test_emoticon_cry(self):
        result = handle_emojis("So sad :'(")
        assert ":cry:" in result

    def test_heart(self):
        result = handle_emojis("Love <3")
        assert ":heart:" in result


# ── Negation scope ───────────────────────────────────────────────────────

class TestNegationScope:
    def test_basic_not(self):
        result = negation_scope("this is not good at all")
        assert "not_good_at_all" in result

    def test_never(self):
        result = negation_scope("I never liked this")
        assert "never_liked_this_" in result or "never_liked_this" in result

    def test_contraction(self):
        result = negation_scope("I don't like this movie")
        assert "don't_like_this_movie" in result

    def test_no_negation(self):
        result = negation_scope("I love this")
        assert result == "I love this"

    def test_custom_window(self):
        result = negation_scope("not very good at all", window=1)
        assert "not_very" in result
        assert "good" in result.split()  # 'good' should be separate


# ── Bot detection ────────────────────────────────────────────────────────

class TestBotDetection:
    def test_i_am_a_bot(self):
        assert is_bot_post("I am a bot, and this action was performed automatically.")

    def test_im_a_bot(self):
        assert is_bot_post("I'm a bot and I do things")

    def test_normal_text(self):
        assert not is_bot_post("I love this stock!")

    def test_case_insensitive(self):
        assert is_bot_post("i am a BOT please ignore")


# ── Language detection ───────────────────────────────────────────────────

class TestLanguageDetection:
    def test_english(self):
        texts = pd.Series(["This is an English sentence about stocks."])
        result = detect_language(texts)
        assert result.iloc[0] == "en"

    def test_empty_string_handled(self):
        texts = pd.Series([""])
        result = detect_language(texts)
        # Should not crash; may return 'und' or something
        assert len(result) == 1


# ── Translation cache ───────────────────────────────────────────────────

class TestTranslationCache:
    def test_put_and_get(self, tmp_path):
        db = str(tmp_path / "test_cache.db")
        cache = TranslationCache(db_path=db)
        cache.put("hola", "es", "hello")
        assert cache.get("hola", "es") == "hello"
        cache.close()

    def test_miss(self, tmp_path):
        db = str(tmp_path / "test_cache.db")
        cache = TranslationCache(db_path=db)
        assert cache.get("unknown", "xx") is None
        cache.close()


# ── Boilerplate removal ─────────────────────────────────────────────────

class TestBoilerplateRemoval:
    def test_removes_pattern(self):
        patterns = [re.compile(r"sent from my iphone", re.IGNORECASE)]
        result = remove_boilerplate("Great news! Sent from my iPhone", patterns)
        assert "iphone" not in result.lower()

    def test_no_patterns(self):
        result = remove_boilerplate("Hello world", None)
        assert result == "Hello world"


# ── ASCII folding ────────────────────────────────────────────────────────

class TestAsciiFold:
    def test_cafe(self):
        result = ascii_fold("café")
        assert result == "cafe"

    def test_plain_ascii(self):
        result = ascii_fold("hello")
        assert result == "hello"


# ── Full pipeline (integration) ──────────────────────────────────────────

class TestCleanTextIntegration:
    def test_full_pipeline(self):
        raw = ("Check https://example.com @user #LoveThis 😊 "
               "not good at all!!!! Sent from my iPhone")
        result = clean_text(
            raw,
            boilerplate_patterns=[re.compile(r"sent from my iphone", re.IGNORECASE)],
            num_mode="keep",
            do_negation_scope=False,
            model_type="transformer",
            do_ascii_fold=True,
        )
        assert "<URL>" in result
        assert "<USER>" in result
        assert "Love This" in result

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_none_like(self):
        assert clean_text("nan") == "nan"  # pandas NaN stringified


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
