"""
Push trained model to hf hub
"""

import huggingface_hub
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


def push_model_to_hf_hub(
        model_id: str,
        checkpoints_dir: str,
        hf_token: str = "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",
):
    ## login to huggingface
    huggingface_hub.login(token=hf_token)
    ## get fine-tuned model config
    config = PeftConfig.from_pretrained(checkpoints_dir)
    ## read base model
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        trust_remote_code=True
    )
    ## read tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name_or_path,
        trust_remote_code=True
    )
    ## read fine-tuned model
    model = PeftModel.from_pretrained(
        model=model,
        model_id=model_id,
        use_auth_token=True,
        is_trainable=True,
    )
    ## push model to hub
    model.push_to_hub(model_id)
    ## push tokenizer
    tokenizer.push_to_hub(model_id)
    ## return model_id
    return {'status': 'successful', 'data': model_id}
