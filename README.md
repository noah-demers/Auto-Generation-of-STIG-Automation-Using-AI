# Auto-Generation-of-STIG-Automation-Using-AI-


Second-semester research developed during my Naval Information Warfare Center internship at the University of Hawaii at Manoa.

This project investigates whether a fine-tuned code-generation model can interpret Security Technical Implementation Guide (STIG) rules, distinguish automatable checks from checks requiring manual review, and generate PowerShell check logic for integration with Evaluate-STIG.

The goal is to reduce the manual scripting burden on cybersecurity engineers while preserving human oversight.

This is a research **prototype**, not a certified compliance tool or an unattended production deployment system.

## Scope

This repository focuses on the second-semester STIG automation project:

- Parsing XCCDF STIG documents.
- Mapping individual rules to existing PowerShell functions.
- Building instruction-based training data.
- Fine-tuning `Qwen2.5-Coder-32B-Instruct`.
- Evaluating classifications and generated PowerShell.
- Assembling generated check logic into Evaluate-STIG-style `.psm1` modules.

The generated code is intended to assess STIG requirements. Generating an assessment check is not the same as automatically remediating a system, proving compliance, or granting authorization to operate.

## Research Objectives

The project explores three questions:

1. Can a model distinguish rules that can be automated from rules requiring manual review?
2. Can it generate useful PowerShell check logic from structured STIG rule content?
3. Can the generated logic be packaged into modules compatible with the expected Evaluate-STIG workflow?

The original research targets were at least 90% coverage of automatable checks and at least 90% functional accuracy. These were research goals, not demonstrated guarantees.

## Pipeline Overview

```text
XCCDF STIG documents + existing PowerShell modules
                         |
                         v
             File-level pairing and XML parsing
                         |
                         v
          Rule-to-function mapping by vulnerability ID
                         |
                         v
              Custom PowerShell code extraction
                         |
                         v
              Instruction-formatted JSONL data
                         |
                         v
             Technology-grouped train/val/test split
                         |
                         v
        Fine-tune Qwen2.5-Coder-32B-Instruct with LoRA
                         |
                         v
          Evaluate classification and code generation
                         |
                         v
          New XCCDF document + fine-tuned adapter
                         |
                         v
       Generate rule-specific code and assemble .psm1
                         |
                         v
            Human review and controlled testing
```

## 1. Parse and Pair the Source Data

The source dataset contains XCCDF STIG documents and corresponding PowerShell `.psm1` modules.

The provided mapping identifies which STIG document corresponds to which PowerShell module. The dataset-building process then establishes the finer rule-to-function relationships.

For each STIG rule, the pipeline extracts information such as:

- Rule and vulnerability identifiers.
- Rule title.
- Severity.
- Description.
- Check text.
- Fix text.

It then matches the rule's vulnerability identifier to the corresponding `Get-V...` function in the PowerShell module.

XML parsing handles the document structure; the language model is used for classification and code generation rather than locating XML fields.

## 2. Extract Rule-Specific PowerShell

Existing Evaluate-STIG functions contain both reusable framework boilerplate and rule-specific check logic.

The dataset-building process extracts the code between the custom-code markers:

```text
Begin Custom Code
...
End Custom Code
```

This replaced an earlier fixed-window extraction approach that could truncate check logic or capture unrelated boilerplate.

Separating custom logic from framework code lets the model focus on the part that changes from one STIG rule to another. The generation pipeline restores the shared module structure afterward.

## 3. Assign Automation Labels

The preparation process distinguishes between rules with an existing mapped implementation and rules without one.

An existing implementation provides evidence that a check has been automated in the reference dataset.

For rules without a mapped implementation, the preparation process uses conservative text heuristics to identify likely automation candidates, likely manual checks, and uncertain cases.

A missing function does not automatically mean that a rule cannot be automated. It may mean that an implementation has not yet been written.

The model's output uses three categories:

| Classification | Meaning |
|---|---|
| `AUTOMATABLE` | The model proposes PowerShell check logic. |
| `MANUAL` | The model identifies the check as requiring manual review. |
| `UNKNOWN` | The model does not confidently determine the automation status. |

Heuristic labels are not equivalent to independently verified ground truth.

## 4. Build Training Examples

The rule records are converted into instruction-based JSONL examples.

Each example contains structured STIG information and a target response combining classification with PowerShell code when applicable.

The expected response structure is:

```json
{
  "classification": "AUTOMATABLE",
  "powershell": "<rule-specific PowerShell code>"
}
```

For manual or uncertain checks:

```json
{
  "classification": "MANUAL",
  "powershell": null
}
```

```json
{
  "classification": "UNKNOWN",
  "powershell": null
}
```

Prompts used during training, evaluation, and inference must remain consistent, including whether the expected PowerShell output is a custom-code body or a complete function.

## 5. Split by Technology

The prepared dataset is split into:

- 70% training.
- 15% validation.
- 15% testing.

The split groups examples by STIG technology or benchmark name rather than randomly distributing individual rules from the same technology across all partitions.

This is intended to reduce leakage between related rules and evaluate generalization to held-out technologies.

Split mappings and audit checks support verification of the dataset partitions.

## 6. Fine-Tune the Model

The project uses:

```text
Qwen/Qwen2.5-Coder-32B-Instruct
```

Training was conducted on the University of Hawaiʻi's Koa high-performance computing environment.

The training approach includes:

- Supervised instruction fine-tuning.
- LoRA adapters instead of full-model parameter updates.
- 4-bit base-model quantization.
- Gradient checkpointing.
- Gradient accumulation.
- Mixed-precision computation.
- Chat-template formatting.
- Padding-token masking in the training labels.
- Saved checkpoints for recovery and later inference.

The fine-tuned adapter is loaded alongside the base model during evaluation and generation.

## 7. Evaluate the Outputs

Evaluation separates classification performance from generated-code quality.

### Classification and Response Validation

The evaluation checks whether outputs:

- Contain parseable JSON.
- Use an allowed classification.
- Follow the expected response schema.
- Include PowerShell when required.
- Leave PowerShell empty when the classification is manual or unknown.

A valid response format does not prove that the generated check is correct.

### PowerShell Evaluation

An early evaluation used strict normalized exact matching against reference code. This proved too restrictive for assessing alternative implementations.

The later evaluation used multiple signals:

| Signal | Purpose |
|---|---|
| Tree-sitter PowerShell parsing and AST-based comparison | Assess syntax and structural similarity. |
| crystalBLEU | Compare code while reducing the influence of common patterns. |
| String overlap | Compare important textual details with reference implementations. |

These metrics are proxies for code quality. Parsing and similarity alone do not establish behavioral equivalence, security, or correctness on a target system.

## Results and Interpretation

The project report records a later evaluation score of 49.97%, described there as functional accuracy, below the original 90% target.

Because the evaluation relied on structural and similarity-based signals, that figure should not be interpreted as a verified runtime success rate across target systems.

Subject-matter expert feedback identified conservative behavior in the generated outputs and suggested that strict evaluation criteria could understate their practical usefulness.

That feedback supports further investigation, but it does not replace execution-based testing or demonstrate that the original target was met.

## 8. Generate an Evaluate-STIG Module

The generation pipeline accepts:

- An XCCDF STIG document.
- A fine-tuned LoRA adapter directory.
- An output `.psm1` path.

It then:

1. Parses the STIG metadata and rules.
2. Reads the base-model identity from the adapter configuration.
3. Loads the base model and adapter.
4. Builds a structured prompt for each rule.
5. Generates a classification and, where applicable, PowerShell check logic.
6. Wraps the custom code in Evaluate-STIG-style function boilerplate.
7. Writes the assembled `.psm1` module.

The assembly includes rule metadata, framework parameters, result-reporting logic, and custom-code markers.

Manual and unknown classifications produce `NotReviewed` placeholders rather than fabricated automated findings.

Malformed model responses fall back to an unknown classification in the generation implementation.

### Generation Arguments

The generation implementation exposes:

| Argument | Purpose |
|---|---|
| `--xccdf` | Input XCCDF XML document. |
| `--adapter` | Fine-tuned LoRA adapter directory. |
| `--out` | Destination `.psm1` file. |
| `--hf-token-from-env` | Explicitly obtain authentication from supported environment variables. |
| `--hf-token-prompt` | Request authentication through a hidden interactive prompt. |

The generated module requires the appropriate Evaluate-STIG framework environment. It is not necessarily a standalone PowerShell script.

## Reproduction Requirements

Reproducing the complete workflow requires more than the source code:

- Source XCCDF documents and reference PowerShell modules.
- File-pairing information.
- Prepared JSONL examples and split manifests.
- Training configuration.
- Base-model access.
- Fine-tuned adapter weights.
- Evaluation tooling, including the PowerShell grammar where required.
- Suitable GPU resources.
- The Evaluate-STIG environment for integration testing.

The implementation uses Python tooling including PyTorch, Hugging Face Transformers, Datasets, PEFT, TRL, and bitsandbytes. The later evaluation additionally uses PowerShell parsing and code-similarity tooling.

Paths and HPC job settings must be adjusted for the environment in which the scripts are run.

## Limitations

- Rules without existing implementations may have heuristic rather than expert-verified labels.
- Held-out technology evaluation does not guarantee reliable performance on every unseen benchmark.
- Generated PowerShell can contain incorrect assumptions, commands, paths, or compliance interpretations.
- Structural similarity does not prove that a generated check behaves like the reference implementation.
- Successful module assembly does not establish compatibility with every Evaluate-STIG version.
- Human review remains necessary for ambiguous rules and generated code.
- Automated remediation and unattended production execution are outside the demonstrated scope.

## Future Work

- Execute generated checks against controlled target-system fixtures.
- Expand expert-reviewed labels for rules without reference implementations.
- Measure false automation decisions separately from missed automation opportunities.
- Improve evaluation of behaviorally equivalent PowerShell implementations.
- Test integration with the complete Evaluate-STIG workflow.
- Evaluate coverage and performance across additional STIG technologies.
- Measure actual reductions in engineer review and scripting time.

## Security and Data Handling

- Do not commit access tokens, private keys, credentials, or sensitive system information.
- Remove hard-coded authentication values from development scripts before publication.
- Publish datasets and reference modules only when redistribution is authorized.
- Review generated code before execution.
- Test generated checks in an isolated, authorized environment.
- Treat generated findings as evidence requiring validation, not as automatic certification of compliance.

## Credits

- Noah deMers — dataset preparation, model fine-tuning, evaluation, and generation pipeline.
- Dr. Curtis Arnold — subject-matter expert feedback on evaluation results.
- William Littleton — faculty advisor.

Developed as part of the second-semester Core4ce / ACE Cyber research work at the University of Hawaiʻi at Mānoa.
