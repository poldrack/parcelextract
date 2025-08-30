PRD prompt:
Help me create a Project Requirement Document (PRD) for a Python module called parcelextract that will take in a 4-dimensional Nifti brain image and extract signal from clusters defined by a specified brain parcellation, saving it to a text file accompanied by a json sidecar file containing relevant metadata. The tool should leverage existing packages such as nibabel, nilearn, and templateflow, and should follow the BIDS standard for file naming as closely as possible. The code should be written in a clean and modular way, using a test-driven development framework.

CLAUDE.md prompt:
Generate a CLAUDE.md file from the attached PRD that will guide Claude Code sessions on this project.  Add the following additional guidelines:

## Development strategy

- Use a test-driven development strategy, developing tests prior to generating solutions to the tests.
- Run the tests and ensure that they fail prior to generating any solutions.
- Write code that passes the tests.
- IMPORTANT: Do not modify the tests simply so that the code passes. Only modify the tests if you identify a specific error in the test.

## Notes for Development

- Think about the problem before generating code.
- Always add a smoke test for the main() function.
- Prefer reliance on widely used packages (such as numpy, pandas, and scikit-learn); avoid unknown packages from Github.
- Do not include any code in init.py files.
- Use pytest for testing.
- Write code that is clean and modular. Prefer shorter functions/methods over longer ones.
- Use functions rather than classes for tests. Use pytest fixtures to share resources between tests.

## Session Guidelines

- Always read PLANNING.md at the start of every new conversation
- Check TASKS.md and SCRATCHPAD.md before starting your work
- Mark completed tasks immediately within TASKS.md
- Add newly discovered tasks to TASKS.md
- use SCRATCHPAD.md as a scratchpad to outline plans

PLANNING.md prompt:

Based on the attached CLAUDE.md and PRD.md files, create a PLANNING.md file that includes  architecture, technology stack, development processes/workflow, and required tools list for this app.

Followup after I saw that it waas using flake8/black:
use ruff for formatting and style checking instead of flake8/black

TASKS.md prompt:

Based on the attached CLAUDE.md and PRD.md files, create a TASKS.md file with buillet points tasks divided into milestones for building this app.

