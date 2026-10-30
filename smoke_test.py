"""CPU smoke test of pid_lib with a TINY RANDOM Gemma 3 (no real weights).

It proves the notebook's code paths run end to end: tokenizer lengths,
fit, evaluate, generate, diagnosis parsing, checkpoint selection,
save/load of LoRA weights. Its scores are meaningless (random model).

    python smoke_test.py
"""
import os
import sys
import tempfile

os.environ.setdefault("KERAS_BACKEND", "jax")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import keras  # noqa: E402
import keras_hub  # noqa: E402
import sentencepiece as spm  # noqa: E402

import pid_lib as L  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def tiny_tokenizer(texts, tmp):
    corpus = os.path.join(tmp, "corpus.txt")
    with open(corpus, "w") as f:
        f.write("\n".join(texts))
    prefix = os.path.join(tmp, "tiny")
    spm.SentencePieceTrainer.train(
        input=corpus, model_prefix=prefix, vocab_size=600, model_type="bpe",
        pad_id=0, unk_id=3, bos_id=1, eos_id=2,
        pad_piece="<pad>", unk_piece="<unk>", bos_piece="<bos>", eos_piece="<eos>",
        user_defined_symbols=["<start_of_turn>", "<end_of_turn>", "<img>",
                              "<start_of_image>", "<end_of_image>"],
        split_digits=True, byte_fallback=True, minloglevel=2)
    return keras_hub.tokenizers.Gemma3Tokenizer(proto=prefix + ".model"), prefix + ".model"


def build_model(tok, seq_len, lr, seed=0, proto=None):
    # Seed FIRST: set_random_seed resets TensorFlow caches, which breaks a
    # tokenizer created before it (the notebook also seeds before loading Gemma).
    keras.utils.set_random_seed(seed)            # same base weights every build
    # a new "session": its own tokenizer object, as in a fresh Colab runtime
    tok = keras_hub.tokenizers.Gemma3Tokenizer(proto=proto)
    backbone = keras_hub.models.Gemma3Backbone(
        vocabulary_size=tok.vocabulary_size(), image_size=896, num_layers=2,
        num_query_heads=2, num_key_value_heads=1, hidden_dim=32, intermediate_dim=64,
        head_dim=16, vision_encoder=None)
    pre = keras_hub.models.Gemma3CausalLMPreprocessor(tokenizer=tok, sequence_length=seq_len)
    model = keras_hub.models.Gemma3CausalLM(backbone=backbone, preprocessor=pre)
    model.backbone.enable_lora(rank=4)
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=keras.optimizers.Adam(learning_rate=lr),
                  weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
                  sampler="greedy")
    return model


def resume_test(tok, seq_len, train, val, tmp, proto):
    """2 epochs straight must equal 1 epoch + 'disconnect' + resume 1 epoch."""
    import numpy as np
    lora_w = lambda m: [np.array(v.value) for v in m.backbone.trainable_weights if "lora" in v.path]
    a = build_model(tok, seq_len, 1e-2, proto=proto)
    ha, _ = L.train_with_selection(a, train, val, 2, os.path.join(tmp, "a_best.lora.h5"), seq_len,
                                   watch_idx=(), gen_batch=4, ckpt_dir=os.path.join(tmp, "ckA"), log=lambda *x: None)
    b = build_model(tok, seq_len, 1e-2, proto=proto)
    L.train_with_selection(b, train, val, 2, os.path.join(tmp, "b_best.lora.h5"), seq_len, watch_idx=(),
                           gen_batch=4, ckpt_dir=os.path.join(tmp, "ckB"), stop_after_epoch=1, log=lambda *x: None)
    del b                                         # the "disconnect": the session is gone
    c = build_model(tok, seq_len, 1e-2, seed=0, proto=proto)  # a NEW session rebuilds the model
    msgs = []
    hc, _ = L.train_with_selection(c, train, val, 2, os.path.join(tmp, "b_best.lora.h5"), seq_len, watch_idx=(),
                                   gen_batch=4, ckpt_dir=os.path.join(tmp, "ckB"), log=msgs.append)
    assert any("RESUMING after epoch 1" in m for m in msgs), "did not resume"
    assert [h["epoch"] for h in hc] == [1, 2], hc
    diff = max(float(np.max(np.abs(x - y))) for x, y in zip(lora_w(a), lora_w(c)))
    print(f"resume test: max |LoRA weight difference| straight vs resumed = {diff:.2e}; "
          f"train loss epoch 2: {ha[1]['train_loss']:.6f} vs {hc[1]['train_loss']:.6f}")
    assert diff < 1e-5, "resumed run differs from the uninterrupted run"
    assert abs(ha[1]["train_loss"] - hc[1]["train_loss"]) < 1e-5
    # the resumed optimizer must be the saved one, not a fresh Adam: its step count is 2 epochs of steps
    assert int(c.optimizer.iterations.value) == int(a.optimizer.iterations.value)
    print("RESUME TEST PASS")


def main():
    keras.utils.set_random_seed(0)
    train = L.load_split(DATA, "train")
    val = L.load_split(DATA, "val")
    # small subsets: this only checks that the code runs
    train = {k: v[:6] for k, v in train.items()}
    val = {k: v[:5] for k, v in val.items()}
    with tempfile.TemporaryDirectory() as tmp:
        tok, proto = tiny_tokenizer(train["prompts"] + train["responses"] + val["prompts"], tmp)
        lens = L.token_lengths(tok, train["prompts"], train["responses"])
        seq_len = L.choose_sequence_length(lens)
        print("token lengths (tiny tokenizer):", min(lens), max(lens), "-> sequence_length", seq_len)

        model = build_model(tok, seq_len, float(os.environ.get("SMOKE_LR", "5e-5")), proto=proto)
        lora = os.path.join(tmp, "best.lora.h5")
        import numpy as np
        lora_vars = [v for v in model.backbone.trainable_weights if "lora" in v.path]
        assert lora_vars, "no trainable LoRA variables found"
        frozen_n = len(model.backbone.non_trainable_weights)
        before = [np.array(v.value) for v in lora_vars]
        hist, best = L.train_with_selection(model, train, val, epochs=int(os.environ.get("SMOKE_EPOCHS", "2")), lora_path=lora,
                                            max_length=seq_len, watch_idx=(0,), gen_batch=4)
        assert os.path.exists(lora), "LoRA weights were not saved"
        # LoRA must actually train: ASSERTED, not just printed
        losses = [h["train_loss"] for h in hist]
        print("train losses:", [round(x, 4) for x in losses])
        changed = sum(not np.allclose(b0, np.array(v.value)) for b0, v in zip(before, lora_vars))
        print(f"LoRA variables changed: {changed}/{len(lora_vars)}; frozen (non-trainable) weights: {frozen_n}")
        assert changed > 0, "no LoRA matrix changed during training"
        assert frozen_n > 0, "base weights are not frozen"
        if os.environ.get("SMOKE_EXPECT_LEARNING") == "1":
            assert losses[-1] < losses[0], f"training loss did not fall: {losses}"
        model.backbone.load_lora_weights(lora)
        s, answers, preds = L.evaluate_split(model, val, seq_len, 4)
        p = L.strip_prompt(val["prompts"][0] + "<start_of_turn>model\nDiagnosis: Well tuned.<end_of_turn>", val["prompts"][0])
        assert L.extract_label(p) == "well_tuned"
        print("best:", best)
        if os.environ.get("SMOKE_RESUME") == "1":
            resume_test(tok, seq_len, train, val, tmp, proto)
        print("SMOKE TEST PASS (code paths ran; scores meaningless with a random model)")


if __name__ == "__main__":
    main()
