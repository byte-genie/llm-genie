"""
Push trained model to hf hub
"""

import huggingface_hub
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


def push_model_to_hf_hub(
        model_id: str,
        checkpoints_dir: str = '/data/checkpoints',
        org_id: str = 'ESGenie',
        private: bool = True,
        merge_base_and_lora: int = 0,
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
        model_id=checkpoints_dir,
        use_auth_token=True,
        is_trainable=True,
    )
    ## if we are to merge base and lora
    if merge_base_and_lora:
        ## merge lora with base
        model = model.merge_and_unload()
    ## push model to hub
    model.push_to_hub(
        f"{org_id}/{model_id}",
        private=private,
        use_auth_token=True,
    )
    ## push tokenizer
    tokenizer.push_to_hub(
        f"{org_id}/{model_id}",
        private=private,
        use_auth_token=True,
    )
    ## return model_id
    return {'status': 'successful', 'data': model_id}


if __name__ == "__main__":
    import argparse
    ## Create the parser
    arg_parser = argparse.ArgumentParser(
        prog='llm-genie',
        usage='%(prog)s [options]',
        description='LLM-Genie'
    )
    ## add arg: model_id
    arg_parser.add_argument('-mid', '--model_id',
                            metavar='model_id',
                            nargs='?',
                            type=str,
                            help='model id')
    ## add arg: checkpoints_dir
    arg_parser.add_argument('-cdir', '--checkpoints_dir',
                            metavar='checkpoints_dir',
                            nargs='?',
                            type=str,
                            default='data/checkpoints',
                            help='local checkpoints directory')
    ## add arg: HF org id
    arg_parser.add_argument('-hf_org', '--hf_org',
                            metavar='hforg',
                            nargs='?',
                            type=str,
                            default='ESGenie',
                            help='HuggingFace organisation id')
    ## add arg: HF token (with write permissions)
    arg_parser.add_argument('-hf_token', '--hf_token',
                            metavar='hf_token',
                            nargs='?',
                            type=str,
                            default='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',
                            help='HuggingFace token with write permissions')
    ## parse args
    args, unknown_args = arg_parser.parse_known_args()
    ## push model to hub
    msg = push_model_to_hf_hub(
        model_id=args.model_id,
        checkpoints_dir=args.checkpoints_dir,
        org_id=args.hf_org,
        hf_token=args.hf_token,
    )
    print(f"msg: {msg}")