import io
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import gradio as gr

import do_detox
from modules import utils

os.environ["COLUMNS"] = "150"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ==========================================
# 工具函数
# ==========================================
class WebLogStream(io.StringIO):
    def __init__(self, max_lines=1000):
        super().__init__()
        self.lines = [""]
        self.max_lines = max_lines

    def write(self, text):
        parts = text.split("\r")
        for i, part in enumerate(parts):
            if i > 0:
                self.lines[-1] = ""

            if part:
                sub_lines = part.split("\n")
                for j, s_line in enumerate(sub_lines):
                    self.lines[-1] += s_line
                    if j < len(sub_lines) - 1:
                        self.lines.append("")

        if len(self.lines) > self.max_lines:
            self.lines = self.lines[-self.max_lines :]

    def flush(self):
        pass

    def getvalue(self):
        return "\n".join(self.lines)


def parse_kv_args(keys, vals):
    """将动态生成的两列文本框转换为字典格式"""
    res = {}
    for k, v in zip(keys, vals):
        if not k or k.strip() == "":
            continue
        k = k.strip()
        v = str(v).strip()

        v_lower = v.lower()
        if v_lower == "true":
            v = True
        elif v_lower == "false":
            v = False
        else:
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
        res[k] = v
    return res


def create_dynamic_kv_ui(defaults, max_rows=30):
    """生成第一列为参数名、第二列为参数值、末尾有加号的表单"""
    keys, vals, rows = [], [], []

    with gr.Column():
        for i in range(max_rows):
            is_visible = i < len(defaults)
            with gr.Row(visible=is_visible) as r:
                k_val = defaults[i][0] if is_visible else ""
                v_val = defaults[i][1] if is_visible else ""
                k = gr.Textbox(
                    value=k_val,
                    placeholder="参数名 (如 learning_rate, 留空则自动忽略)",
                    show_label=False,
                    container=False,
                    scale=1,
                )
                v = gr.Textbox(
                    value=v_val,
                    placeholder="参数值 (如 1e-5)",
                    show_label=False,
                    container=False,
                    scale=1,
                )
            rows.append(r)
            keys.append(k)
            vals.append(v)

        add_btn = gr.Button("➕ 新增一行参数", size="sm")
        visible_count = gr.State(len(defaults))

        def add_one_row(count):
            new_count = min(count + 1, max_rows)
            updates = [gr.update(visible=(i < new_count)) for i in range(max_rows)]
            return [new_count] + updates

        add_btn.click(add_one_row, inputs=[visible_count], outputs=[visible_count] + rows)

    return keys, vals


def _safe_timestamp():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _normalize_label_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip().lower()


def _save_uploaded_jsonl(upload_path):
    if not upload_path:
        raise ValueError("请先上传 jsonl 文件。")

    src_name = os.path.basename(upload_path)
    if not src_name.lower().endswith(".jsonl"):
        raise ValueError("仅支持上传 .jsonl 文件。")

    os.makedirs("data/users", exist_ok=True)
    name_wo_ext = os.path.splitext(src_name)[0]
    dst_name = f"{name_wo_ext}_{_safe_timestamp()}.jsonl"
    dst_path = os.path.join("data/users", dst_name)
    shutil.copyfile(upload_path, dst_path)
    return dst_path


def _build_seqcls_dataset(input_jsonl, text_field, label_field, positive_label_value):
    if not os.path.exists(input_jsonl):
        raise FileNotFoundError(f"输入文件不存在: {input_jsonl}")

    os.makedirs("data/users/processed", exist_ok=True)
    base = os.path.splitext(os.path.basename(input_jsonl))[0]
    out_path = os.path.join("data/users/processed", f"{base}_seqcls_{_safe_timestamp()}.jsonl")

    pos_norm = _normalize_label_value(positive_label_value)
    if pos_norm == "":
        raise ValueError("正类取值不能为空。")

    total = 0
    valid = 0
    skipped = 0
    label_0 = 0
    label_1 = 0

    with open(input_jsonl, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            line = line.strip()
            if not line:
                skipped += 1
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            if text_field not in obj or label_field not in obj:
                skipped += 1
                continue

            text = str(obj.get(text_field, "")).strip()
            if text == "":
                skipped += 1
                continue

            label_norm = _normalize_label_value(obj.get(label_field))
            label = 1 if label_norm == pos_norm else 0

            if label == 1:
                label_1 += 1
            else:
                label_0 += 1

            swift_record = {
                "messages": [{"role": "user", "content": text}],
                "label": label,
            }
            fout.write(json.dumps(swift_record, ensure_ascii=False) + "\n")
            valid += 1

    if valid == 0:
        raise ValueError("预处理后没有可用样本，请检查字段名与数据内容。")
    if label_0 == 0 or label_1 == 0:
        raise ValueError("预处理后仅包含单一类别，请检查正类取值是否正确。")

    summary = (
        f"预处理完成: {out_path}\n"
        f"总行数={total}, 有效样本={valid}, 跳过样本={skipped}, label0={label_0}, label1={label_1}"
    )
    return out_path, summary


# ==========================================
# 页面1: 上传/预处理/二分类训练
# ==========================================
def ui_save_user_dataset(uploaded_file):
    log_stream = WebLogStream(max_lines=500)
    original_stdout, original_stderr = sys.stdout, sys.stderr

    def task():
        sys.stdout = sys.stderr = log_stream
        try:
            saved = _save_uploaded_jsonl(uploaded_file)
            print(f"Uploaded dataset saved to: {saved}")
            return f"上传成功: {saved}", saved, saved, saved
        except Exception as e:
            return f"上传失败: {str(e)}", None, None, None
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(task)

    while not future.done():
        time.sleep(0.2)
        yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), log_stream.getvalue()

    status, saved_path, upload_dl, train_dataset = future.result()
    yield status, saved_path, upload_dl, train_dataset, log_stream.getvalue()


def ui_prepare_seqcls_dataset(saved_dataset_path, text_field, label_field, positive_label):
    log_stream = WebLogStream(max_lines=500)
    original_stdout, original_stderr = sys.stdout, sys.stderr

    def task():
        sys.stdout = sys.stderr = log_stream
        try:
            out_path, summary = _build_seqcls_dataset(
                input_jsonl=saved_dataset_path,
                text_field=text_field,
                label_field=label_field,
                positive_label_value=positive_label,
            )
            print(summary)
            return summary, out_path, out_path, out_path
        except Exception as e:
            return f"预处理失败: {str(e)}", None, None, None
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(task)

    while not future.done():
        time.sleep(0.5)
        yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), log_stream.getvalue()

    status, processed_path, processed_dl, train_dataset = future.result()
    yield status, processed_path, processed_dl, train_dataset, log_stream.getvalue()


def ui_run_seqcls_train(seed, model_path, train_dataset_path, train_output_dir, *kv_args):
    log_stream = WebLogStream(max_lines=500)
    original_stdout, original_stderr = sys.stdout, sys.stderr

    def task():
        sys.stdout = sys.stderr = log_stream
        try:
            if not train_dataset_path or not os.path.exists(train_dataset_path):
                raise ValueError("训练数据集路径无效，请先上传并预处理。")

            os.makedirs(train_output_dir, exist_ok=True)

            ft_keys = kv_args[:30]
            ft_vals = kv_args[30:60]
            ft_kwargs = parse_kv_args(ft_keys, ft_vals)
            ft_kwargs["task_type"] = "seq_cls"
            if "num_labels" not in ft_kwargs:
                ft_kwargs["num_labels"] = 2

            result = utils.finetune_model_with_swift(
                seed=int(seed),
                model_id_or_path=model_path,
                dataset_path=train_dataset_path,
                output_dir=train_output_dir,
                **ft_kwargs,
            )

            best_ckpt = result.get("best_model_checkpoint") if isinstance(result, dict) else None
            last_ckpt = result.get("last_model_checkpoint") if isinstance(result, dict) else None
            status = "二分类训练完成。"
            print(status)
            print(f"Best checkpoint: {best_ckpt}")
            print(f"Last checkpoint: {last_ckpt}")
            return status, best_ckpt, last_ckpt, result
        except Exception as e:
            return f"训练失败: {str(e)}", None, None, {"error": str(e)}
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(task)

    while not future.done():
        time.sleep(1.0)
        yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), log_stream.getvalue()

    status, best_ckpt, last_ckpt, result_json = future.result()
    yield status, best_ckpt, last_ckpt, result_json, log_stream.getvalue()


# ==========================================
# 页面2: 文本解毒 (完整复用 app_ui.py detox)
# ==========================================
def ui_run_detox(
    loop_idx,
    seed,
    prefix,
    steps,
    s1_mpath,
    s1_data,
    s1_data_field,
    s1_out,
    use_new_toxic,
    tox_path,
    s2_in,
    s2_text_field,
    s2_work,
    s2_mtype,
    s2_msize,
    s2_mpath,
    s2_bs,
    s2_mpl,
    s2_mnt,
    s2_mgr,
    s2_ds,
    s2_temp,
    s2_shots,
    detox_mode,
    diffs_fn,
    wsd_p,
    wsd_blur,
    v_alpha,
    v_b1,
    v_b2,
    tk_num,
    tk_b1,
    tk_b2,
    socd_kset,
    socd_kmax,
    socd_minv,
    socd_b1,
    socd_b2,
    s3_in,
    s3_em_mod,
    s3_em_path,
    s3_dim,
    s3_dtx_type,
    s3_dtx_ckpt,
    s3_dtx_hf,
    s3_rm_name,
    s3_rm_lam,
    *kv_args,
):
    log_stream = WebLogStream(max_lines=500)
    original_stdout, original_stderr = sys.stdout, sys.stderr

    def task():
        sys.stdout = sys.stderr = log_stream
        try:
            diffs_fn_args = {"wsd_p": int(wsd_p), "wsd_blur": float(wsd_blur)} if diffs_fn == "wsd" else {}

            s1_ft_keys = kv_args[:30]
            s1_ft_vals = kv_args[30:60]
            s1_ft_dict = parse_kv_args(s1_ft_keys, s1_ft_vals)

            pipe_config = {
                "loop_idx": int(loop_idx),
                "seed": int(seed),
                "output_dir_prefix": prefix,
                "run_step_1": "有毒小模型微调" in steps,
                "run_step_2": "文本解毒" in steps,
                "run_step_3": "融合排序" in steps,
                "step1_config": {
                    "model_path": s1_mpath,
                    "dataset": s1_data,
                    "text_field": s1_data_field,
                    "output_dir": s1_out,
                    "train_args": s1_ft_dict,
                },
                "step2_config": {
                    "use_new_toxic_model": not (use_new_toxic),
                    "data": {
                        "input_file": s2_in,
                        "text_field": s2_text_field,
                        "data_workers": int(s2_work),
                    },
                    "model": {
                        "model_type": s2_mtype,
                        "model_size_info": s2_msize,
                        "model_path": s2_mpath,
                        "toxic_model_path": tox_path,
                    },
                    "gen": {
                        "batch_size": int(s2_bs),
                        "max_prompt_length": int(s2_mpl),
                        "max_new_tokens": int(s2_mnt),
                        "max_gen_repeat_times": int(s2_mgr),
                        "do_sample": s2_ds,
                        "temperature_list": [float(x.strip()) for x in s2_temp.split(",")],
                        "detox_mode_name": detox_mode,
                    },
                    "detox": {
                        "diffs_fn": {"diffs_fn_name": diffs_fn, "diffs_fn_args": diffs_fn_args},
                        "shots_num": int(s2_shots),
                        "vanilla_cd": {"alpha": float(v_alpha), "beta1": float(v_b1), "beta2": float(v_b2)},
                        "top_k_cd": {"top_k_num": int(tk_num), "beta1": float(tk_b1), "beta2": float(tk_b2)},
                        "socd": {
                            "k_setting": socd_kset,
                            "k_max": float(socd_kmax),
                            "min_vocab": int(socd_minv),
                            "beta1": float(socd_b1),
                            "beta2": float(socd_b2),
                        },
                    },
                },
                "step3_config": {
                    "data": {"input_file": s3_in},
                    "embeddings": {
                        "embedding_model": s3_em_mod,
                        "embedding_model_path": s3_em_path,
                        "dim_num": int(s3_dim),
                    },
                    "detoxify": {
                        "model_type": s3_dtx_type,
                        "ckpt_path": s3_dtx_ckpt,
                        "huggingface_config_path": s3_dtx_hf,
                    },
                    "rank_mode": {"rank_mode_name": s3_rm_name, "lambda_value": float(s3_rm_lam)},
                },
            }

            do_detox.run_detox_pipeline(pipe_config)

            out_dir = f"{prefix}_loop_{loop_idx}"
            detox_file = os.path.join(out_dir, f"detox_{detox_mode}_{s2_mtype}_{s2_msize}.jsonl")
            base_name = os.path.splitext(os.path.basename(s3_in if "融合排序" not in steps else detox_file))[0]
            rank_file = os.path.join(out_dir, f"ranked_{s3_rm_name}_{base_name}.jsonl")

            new_loop_idx = loop_idx + 1
            new_dataset_path = rank_file if "融合排序" in steps else detox_file

            return new_loop_idx, new_loop_idx, new_dataset_path, "detoxified_texts", detox_file, rank_file
        except Exception as e:
            return loop_idx, loop_idx, str(e), "text", None, None
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(task)

    while not future.done():
        time.sleep(1.0)
        yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), log_stream.getvalue()

    r_loop, r_disp, r_ds, r_field, r_detox_dl, r_rank_dl = future.result()
    yield r_loop, r_disp, r_ds, r_field, r_detox_dl, r_rank_dl, log_stream.getvalue()


# ==========================================
# UI 构建
# ==========================================
with gr.Blocks(title="二分类训练", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🧪 自定义二分类训练")

    loop_idx_state = gr.State(value=0)

    with gr.Row(variant="panel"):
        seed_input = gr.Number(value=1337, label="全局随机种子", precision=0)
        # loop_idx_display = gr.Number(value=0, label="当前解毒迭代次数", interactive=False)

    with gr.Tabs():
        with gr.Tab("自定义数据集二分类训练"):
            with gr.Tabs():
                with gr.Tab("第一部分. 上传与预处理"):
                    gr.Markdown("### 1) 上传用户自定义 JSONL 数据集")
                    upload_file = gr.File(label="上传 JSONL 文件", file_types=[".jsonl"], type="filepath")
                    upload_btn = gr.Button("📤 保存上传文件到 data/users", variant="primary")
                    upload_status = gr.Textbox(label="上传状态", interactive=False)
                    saved_dataset_path = gr.Textbox(label="已保存数据集路径", interactive=False)
                    upload_download = gr.File(label="已保存文件下载")

                    gr.Markdown("### 2) 二分类预处理")
                    with gr.Row():
                        cls_text_field = gr.Textbox(value="text", label="文本字段名")
                        cls_label_field = gr.Textbox(value="label", label="标签字段名")
                        cls_positive_value = gr.Textbox(value="1", label="正类取值(映射为1)")
                    preprocess_btn = gr.Button("🧹 运行二分类预处理", variant="primary")
                    preprocess_status = gr.Textbox(label="预处理状态", interactive=False)
                    processed_dataset_path = gr.Textbox(label="预处理后数据集路径", interactive=False)
                    preprocess_download = gr.File(label="预处理后文件下载")

                with gr.Tab("第二部分. 二分类任务训练"):
                    gr.Markdown("### 基于 ms-swift 的 seq_cls 微调")
                    with gr.Row():
                        cls_model_path = gr.Textbox(value="models/Qwen2.5-0.5B-Instruct", label="基础模型路径")
                        cls_dataset_path = gr.Textbox(value="", label="训练数据集路径 (由预处理自动填充)")
                        cls_output_dir = gr.Textbox(value="output/seqcls_model_sft", label="微调输出目录")

                    with gr.Group(visible=True):
                        gr.Markdown("#### 训练参数表 (默认与 detox 中小模型微调一致，task_type=seq_cls)")
                        seqcls_defaults = [
                            ["task_type", "seq_cls"],
                            ["train_type", "full"],
                            ["max_steps", "6"],
                            ["split_dataset_ratio", "0.2"],
                            ["save_steps", "2"],
                            ["eval_steps", "2"],
                            ["per_device_train_batch_size", "1"],
                            ["per_device_eval_batch_size", "1"],
                            ["logging_steps", "1"],
                            ["metric_for_best_model", "eval_acc"],
                            ["greater_is_better", "True"],
                            ["load_best_model_at_end", "True"],
                            ["save_total_limit", "1"],
                            ["use_chat_template", "True"],
                        ]
                        seqcls_ft_keys, seqcls_ft_vals = create_dynamic_kv_ui(seqcls_defaults)

                    seqcls_train_btn = gr.Button("🚀 开始二分类训练", variant="primary")
                    seqcls_train_status = gr.Textbox(label="训练状态", interactive=False)
                    with gr.Row():
                        seqcls_best_ckpt = gr.Textbox(label="best_model_checkpoint", interactive=False)
                        seqcls_last_ckpt = gr.Textbox(label="last_model_checkpoint", interactive=False)
                    seqcls_result_json = gr.JSON(label="训练返回结果")

        # with gr.Tab("2. 文本解毒模块"):
        #     with gr.Row():
        #         detox_out_prefix = gr.Textbox(value="output/detox", label="解毒输出路径")
        #         detox_steps = gr.CheckboxGroup(
        #             choices=["有毒小模型微调", "文本解毒", "融合排序"],
        #             value=["文本解毒", "融合排序"],
        #             label="运行步骤",
        #         )

        #     with gr.Tabs():
        #         with gr.Tab("第一部分. 有毒小模型微调"):
        #             with gr.Row():
        #                 s1_mpath = gr.Textbox(value="models/Qwen2.5-0.5B-Instruct", label="基础模型路径")
        #                 s1_data = gr.Textbox(value="data/dghs_to_detox.jsonl", label="数据集路径")
        #                 s1_data_field = gr.Textbox(value="text", label="文本字段名")
        #                 s1_out = gr.Textbox(value="output/toxic_model_sft", label="微调模型输出目录")

        #             with gr.Group(visible=True):
        #                 gr.Markdown("#### 有毒小模型微调参数表")
        #                 s1_defaults = [
        #                     ["train_type", "full"],
        #                     ["max_steps", "6"],
        #                     ["split_dataset_ratio", "0.2"],
        #                     ["save_steps", "2"],
        #                     ["eval_steps", "2"],
        #                     ["per_device_train_batch_size", "1"],
        #                     ["per_device_eval_batch_size", "1"],
        #                     ["logging_steps", "1"],
        #                     ["metric_for_best_model", "eval_token_acc"],
        #                     ["greater_is_better", "True"],
        #                     ["load_best_model_at_end", "True"],
        #                     ["save_total_limit", "1"],
        #                     ["use_chat_template", "True"],
        #                 ]
        #                 s1_ft_keys, s1_ft_vals = create_dynamic_kv_ui(s1_defaults)

        #         with gr.Tab("二. 文本解毒算法"):
        #             use_new_toxic_cb = gr.Checkbox(label="使用自定义有毒小模型", value=True)
        #             toxic_path_tb = gr.Textbox(value="models/toxic-sft-dghs-qwen25-0.5b", label="有毒小模型路径", visible=True)
        #             use_new_toxic_cb.change(lambda x: gr.update(visible=x), inputs=use_new_toxic_cb, outputs=toxic_path_tb)

        #             with gr.Row():
        #                 s2_in = gr.Textbox(value="data/dghs_to_detox.jsonl", label="被解毒文件")
        #                 s2_text_field = gr.Textbox(value="text", label="被解毒文本字段名")
        #                 s2_mpath = gr.Textbox(value="models/Qwen2.5-3B-Instruct", label="基础模型路径")
        #             with gr.Accordion("推理解码参数", open=True):
        #                 with gr.Row():
        #                     s2_mtype = gr.Textbox(value="qwen2", label="模型类型 (qwen2 或 gpt2)")
        #                     s2_msize = gr.Textbox(value="3-0.5", label="模型尺寸标识")
        #                     s2_shots = gr.Number(value=1, label="解毒提示词的 shots 数")
        #                     s2_work = gr.Number(value=4, label="data_workers")
        #                 with gr.Row():
        #                     s2_ds = gr.Checkbox(value=True, label="do_sample")
        #                 with gr.Row():
        #                     s2_bs = gr.Number(value=2, label="批次大小")
        #                     s2_mpl = gr.Number(value=512, label="最大提示词长度")
        #                     s2_mnt = gr.Number(value=512, label="最大生成长度")
        #                     s2_mgr = gr.Number(value=2, label="重复生成次数")
        #                     s2_temp = gr.Textbox(value="0.7, 1.0", label="采样温度列表 (半角逗号分隔)")

        #             gr.Markdown("#### 解毒核心算法策略")
        #             with gr.Row():
        #                 detox_mode = gr.Dropdown(
        #                     choices=["prompt_only", "vanilla_cd", "top_k_cd", "socd"],
        #                     value="socd",
        #                     label="解毒模式",
        #                 )
        #                 diffs_fn_name = gr.Dropdown(
        #                     choices=["none", "fkl", "rkl", "js", "tvd", "wsd"],
        #                     value="js",
        #                     label="分布差异度量函数",
        #                 )

        #             with gr.Group(visible=False) as wsd_group:
        #                 with gr.Row():
        #                     wsd_p_in = gr.Number(value=1, label="wsd_p")
        #                     wsd_blur_in = gr.Number(value=0.01, label="wsd_blur")
        #             diffs_fn_name.change(lambda m: gr.update(visible=m == "wsd"), inputs=diffs_fn_name, outputs=wsd_group)

        #             with gr.Group(visible=False) as vanilla_cd_group:
        #                 with gr.Row():
        #                     v_alpha = gr.Number(value=0.1, label="alpha")
        #                     v_b1 = gr.Number(value=0.5, label="beta1")
        #                     v_b2 = gr.Number(value=0.5, label="beta2")

        #             with gr.Group(visible=False) as topk_cd_group:
        #                 with gr.Row():
        #                     tk_num = gr.Number(value=10, label="top_k_num")
        #                     tk_b1 = gr.Number(value=0.5, label="beta1")
        #                     tk_b2 = gr.Number(value=0.5, label="beta2")

        #             with gr.Group(visible=True) as socd_group:
        #                 with gr.Row():
        #                     socd_kset = gr.Textbox(value="dynamic", label="k_setting")
        #                     socd_kmax = gr.Number(value=0.5, label="k_max (一般由评估过程自动调控)", interactive=True)
        #                 with gr.Row():
        #                     socd_minv = gr.Number(value=10, label="min_vocab")
        #                     socd_b1 = gr.Number(value=0.5, label="beta1")
        #                     socd_b2 = gr.Number(value=0.5, label="beta2")

        #             detox_mode.change(
        #                 lambda m: (
        #                     gr.update(visible=m == "vanilla_cd"),
        #                     gr.update(visible=m == "top_k_cd"),
        #                     gr.update(visible=m == "socd"),
        #                 ),
        #                 inputs=detox_mode,
        #                 outputs=[vanilla_cd_group, topk_cd_group, socd_group],
        #             )

        #         with gr.Tab("三. 融合排序"):
        #             with gr.Row():
        #                 s3_in = gr.Textbox(value="output/detox_loop_1/detox_socd_qwen2_3-0.5.jsonl", label="解毒后文件路径")
        #             gr.Markdown("#### 嵌入模型配置")
        #             with gr.Row():
        #                 s3_em_mod = gr.Textbox(value="Qwen3-Embedding-0.6B", label="嵌入模型名")
        #                 s3_em_path = gr.Textbox(value="models/Qwen3-Embedding-0.6B", label="嵌入模型路径")
        #                 s3_dim = gr.Number(value=1024, label="嵌入维度")
        #             gr.Markdown("#### Detoxify 模型配置")
        #             with gr.Row():
        #                 s3_dtx_type = gr.Textbox(value="original", label="model_type")
        #                 s3_dtx_ckpt = gr.Textbox(value="models/detoxify/toxic_original-c1212f89.ckpt", label="ckpt_path")
        #                 s3_dtx_hf = gr.Textbox(value="models/bert-base-uncased", label="hf_config")
        #             gr.Markdown("#### 算法与参数配置")
        #             with gr.Row():
        #                 s3_rm_name = gr.Textbox(value="cosine_similarity", label="语义相似度算法")
        #                 s3_rm_lam = gr.Number(value=0.5, label="lambda 参数")

        #         detox_btn = gr.Button("🛡️ 运行解毒流水线", variant="primary")
        #         with gr.Row():
        #             detox_file_dl = gr.File(label="📥 解毒后文件下载")
        #             rank_file_dl = gr.File(label="📥 排序后文件下载")

    gr.Markdown("---")
    gr.Markdown("### 🖥️ 全局运行日志")
    log_output = gr.Textbox(label="实时终端捕获", lines=12, max_lines=25, interactive=False, show_copy_button=True)

    # ==========================================
    # Logic Binding
    # ==========================================
    upload_btn.click(
        ui_save_user_dataset,
        inputs=[upload_file],
        outputs=[upload_status, saved_dataset_path, upload_download, cls_dataset_path, log_output],
    )

    preprocess_btn.click(
        ui_prepare_seqcls_dataset,
        inputs=[saved_dataset_path, cls_text_field, cls_label_field, cls_positive_value],
        outputs=[preprocess_status, processed_dataset_path, preprocess_download, cls_dataset_path, log_output],
    )

    seqcls_train_btn.click(
        ui_run_seqcls_train,
        inputs=[seed_input, cls_model_path, cls_dataset_path, cls_output_dir] + seqcls_ft_keys + seqcls_ft_vals,
        outputs=[seqcls_train_status, seqcls_best_ckpt, seqcls_last_ckpt, seqcls_result_json, log_output],
    )

    # detox_btn.click(
    #     ui_run_detox,
    #     inputs=[
    #         loop_idx_state,
    #         seed_input,
    #         detox_out_prefix,
    #         detox_steps,
    #         s1_mpath,
    #         s1_data,
    #         s1_data_field,
    #         s1_out,
    #         use_new_toxic_cb,
    #         toxic_path_tb,
    #         s2_in,
    #         s2_text_field,
    #         s2_work,
    #         s2_mtype,
    #         s2_msize,
    #         s2_mpath,
    #         s2_bs,
    #         s2_mpl,
    #         s2_mnt,
    #         s2_mgr,
    #         s2_ds,
    #         s2_temp,
    #         s2_shots,
    #         detox_mode,
    #         diffs_fn_name,
    #         wsd_p_in,
    #         wsd_blur_in,
    #         v_alpha,
    #         v_b1,
    #         v_b2,
    #         tk_num,
    #         tk_b1,
    #         tk_b2,
    #         socd_kset,
    #         socd_kmax,
    #         socd_minv,
    #         socd_b1,
    #         socd_b2,
    #         s3_in,
    #         s3_em_mod,
    #         s3_em_path,
    #         s3_dim,
    #         s3_dtx_type,
    #         s3_dtx_ckpt,
    #         s3_dtx_hf,
    #         s3_rm_name,
    #         s3_rm_lam,
    #     ]
    #     + s1_ft_keys
    #     + s1_ft_vals,
    #     outputs=[loop_idx_state, loop_idx_display, s2_in, s2_text_field, detox_file_dl, rank_file_dl, log_output],
    # )


if __name__ == "__main__":
    app.launch(server_name=os.getenv("APP_HOST", "127.0.0.1"), server_port=int(os.getenv("APP_PORT", "17891")), share=False)
