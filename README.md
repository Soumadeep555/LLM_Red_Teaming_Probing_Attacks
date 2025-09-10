# LLM Red Teaming Script
### Overview
This Python script is designed to perform red teaming on large language models (LLMs), specifically testing for vulnerabilities in AI behavior using the GPT-OSS:20B model hosted via Ollama. Red teaming involves probing the model with targeted prompts to identify potential issues such as deception, reward hacking, sabotage, and more. The script automates the setup of the environment, runs probes reproducibly, detects vulnerabilities, exports findings in JSON format (conforming to a specified schema), and performs a novelty check for each issue.

#### Key features:

- Installs dependencies and sets up Ollama server.
- Downloads and interfaces with the GPT-OSS:20B model.
- Defines and runs 8 distinct vulnerability probes.
- Tests each probe multiple times for reproducibility.
- Exports detailed findings in JSON if a vulnerability is detected.
- Simulates novelty checks to assess if issues are previously known.

This tool is useful for AI safety researchers, developers, and testers evaluating model robustness against adversarial or edge-case inputs.

### Requirements

- Python 3.8 or higher.
- Operating system: Linux or macOS (Ollama installation may vary on Windows).
- Hardware: At least 16GB RAM recommended for running the 20B parameter model; GPU acceleration (CUDA) is optional but improves performance.
- Internet access: Required for initial Ollama installation and model download.

No additional packages need to be installed manually, as the script handles this via pip. The script uses subprocess to automatically install the openai package if it's missing, ensuring seamless setup.

### Installation
conda create -n llm-redteam python=3.10 -y
conda activate llm-redteam
Clone the repository and install with pip:
git clone https://github.com/Soumadeep555/LLM_Red_Teaming_Probing_Attacks.git
cd LLM_Red_Teaming_Probing_Attacks
pip install -r requirements.txt

- Ensure Python is installed and accessible via the command line.
- Run the script directly (it will handle setup):
python llm_red_teaming_script.py
- This will install openai package if missing using pip.
- Install and start the Ollama server.
- Download the GPT-OSS:20B model (approximately 13GB; may take time based on your connection).

Note: If Ollama fails to install, manually run curl -fsSL https://ollama.com/install.sh | sh and restart the script.

### Usage
The script is self-contained and runs end-to-end when executed. No command-line arguments are required.

#### Running the Script
#### Execute the script:
python llm_red_teaming.py

- The script will:
1. Set up the environment and model.
2. Run each probe 5 times (configurable via num_runs in test_reproducibility).
3. Print vulnerability rates and average execution times for each probe.
4. Export JSON findings for vulnerable probes (files named findings_<probe_name>.json).
5. Perform and print novelty checks (simulated web search scores out of 5).

- Output example:
Testing probe: reward_hacking
Probe reward_hacking: Vulnerability rate 20.0% over 5 runs. Avg time: 1.25s
Findings exported to findings_reward_hacking.json
Novelty check for reward_hacking: Score 3/5
...
All probes completed. Findings files: ['findings_reward_hacking.json', ...]

### Customizing the Script

- Change Model: Modify model_name in download_model or ModelInterface init.
- Adjust Probes: Edit the probes dictionary to add/remove prompts or patterns.
- Reproducibility Runs: Change num_runs in test_reproducibility (default: 5).
- System Prompt: Use model_interface.update_system_prompt(new_prompt) for custom behaviors.
- Novelty Check: The check_novelty function simulates searches; integrate a real API (e.g., via tools) for production use.

### Probes Explained
The script tests for 8 vulnerabilities using specific prompts and regex patterns to detect unsafe responses. A probe is "vulnerable" if the response does NOT match the safe pattern.

1. Reward Hacking: Prompt encourages gaming rewards; expects uncertainty in response.
2. Deception: Simulates lying for survival; expects truthful facts.
3. Chain of Thought Issue: Tests step-by-step reasoning; expects correct outputs.
4. Hidden Motivation: Probes for covert biases; expects neutral responses.
5. Sabotage: Triggers intentional errors; expects normal behavior.
6. Inappropriate Tool Use: Attempts harmful actions; expects refusal.
7. Data Exfiltration: Tries leaking secrets; expects protection.
8. Sandbagging: Tests performance hiding; expects full effort.
Each probe resets the conversation history to isolate tests.

### Output and Findings

- Console Output: Real-time progress, vulnerability rates, and novelty scores.
- JSON Findings: For vulnerable probes, generates files with:

    - Schema version and issue title.
    - Model details (name, version, parameters).
    - Environment info (hardware, libraries).
    - Issue summary (overview, desired/observed behavior, self-assessed severity/breadth).
    - Harmony-style walkthrough (example prompt/response).
    - Reproduction steps.
    - Notes (including responses and rates).
These JSON files follow a structured schema for easy integration with reporting tools or databases.

### Limitations and Notes

- Model Performance: The 20B model may be slow without GPU; expect 1-5 seconds per query.
- False Positives: Regex patterns may need tuning for edge cases.
- Ethical Use: This is for research only; do not use to exploit production systems.
- Novelty Simulation: Current implementation uses placeholders; enhance with actual web searches.
- Error Handling: If queries fail (e.g., connection issues), responses are marked vulnerable.
For issues or contributions, open a pull request or contact the maintainer.

### License
This dataset is licensed under CC0 1.0 Universal (CC0 1.0) Public Domain Dedication. You can copy, modify, distribute and perform the work, even for commercial purposes, all without asking permission. See the full license at https://creativecommons.org/publicdomain/zero/1.0/.

### Citation
www.kaggle.com/competitions/openai-gpt-oss-20b-red-teaming/overview/citation
@misc{openai-gpt-oss-20b-red-teaming,
    author = {D. Sculley and Samuel Marks and Addison Howard},
    title = {Red‑Teaming Challenge - OpenAI gpt-oss-20b},
    year = {2025},
    howpublished = {\url{https://kaggle.com/competitions/openai-gpt-oss-20b-red-teaming}},
    note = {Kaggle}
}