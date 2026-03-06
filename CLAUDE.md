# NeuroQ Project Configuration

## Plugins

### huggingface-skills (installed from claude-plugins-official)
- **Source**: https://github.com/huggingface/skills
- **Location**: `.claude/plugins/huggingface-skills/`
- **Description**: Agent Skills for AI/ML tasks including dataset creation, model training, evaluation, and research paper publishing on Hugging Face Hub

#### Available Skills

| Skill | Description |
|-------|-------------|
| `hugging-face-model-trainer` | Train/fine-tune LLMs using TRL on HF Jobs (SFT, DPO, GRPO, reward modeling, GGUF conversion) |
| `hugging-face-datasets` | Create and manage datasets on HF Hub (init repos, streaming updates, SQL queries) |
| `hugging-face-evaluation` | Add evaluation results to model cards, run custom evals with vLLM/lighteval |
| `hugging-face-jobs` | Run compute jobs on HF infrastructure (UV scripts, Docker jobs) |
| `hugging-face-trackio` | Track ML experiments with Trackio (log metrics, real-time dashboards) |
| `hugging-face-paper-publisher` | Publish research papers on HF Hub |
| `hugging-face-tool-builder` | Build reusable scripts for HF API operations |
| `hugging-face-dataset-viewer` | Explore datasets via Dataset Viewer REST API |
| `hf-cli` | HF Hub operations via CLI (download, upload, manage repos) |
| `gradio` | Build Gradio web UIs and Python demos |

#### Usage

When a task matches a skill's description, read the corresponding `SKILL.md` file from `.claude/plugins/huggingface-skills/skills/<skill-name>/SKILL.md` for detailed instructions, scripts, and reference materials.

Skills include executable scripts (Python/Shell) in their `scripts/` subdirectories, reference documentation in `references/`, and templates in `templates/` where applicable.

#### Prerequisites

- `HF_TOKEN` environment variable for authenticated Hub operations
- `uv` package manager for running PEP 723 inline-dependency scripts
- Python 3.8+ with `pip` or `uv` for dependency management
