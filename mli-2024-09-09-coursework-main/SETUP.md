# Using this repo

1. [Initializing your environment](#initializing-your-environment)
   1. [Common Issues](#common-issues)
      1. ["I don't see the interpreter in the list"](#i-dont-see-the-interpreter-in-the-list)
      2. ["I need to install different dependencies"](#i-need-to-install-different-dependencies)
2. [Submitting Your Work](#submitting-your-work)
3. [Fetching updates](#fetching-updates)
   1. [Workflow 1: Simple Merge](#workflow-1-simple-merge)
   2. [Workflow 2: Rebase](#workflow-2-rebase)
4. [Running notebooks on other platforms](#running-notebooks-on-other-platforms)

> NOTE: The instructions below presume you have completed the environment setup steps in detailed in Precourse and have these tools installed:
>
> - [Python v3.11](https://www.python.org/downloads/release/python-3110/)
> - [Pipenv](https://pipenv.pypa.io/en/latest/install/#installing-pipenv) (environment and package manager)
> - An IDE: [VS Code](https://code.visualstudio.com/download) or [Cursor](https://cursor.sh), with the following extensions:
>   - [Github Copilot](https://marketplace.visualstudio.com/items?itemName=GitHub.copilot) and [Chat](https://marketplace.visualstudio.com/items?itemName=GitHub.copilot-chat)
>   - [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
>   - [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
>
> Reach out to staff if you run into issues setting up your environment.

## Initializing your environment

- [ ] Clone this repo:
  - e.g. `git clone https://github.com/deepatlasai/mli-0000-00-coursework.git`
- [ ] Navigate into the cloned directory:
  - e.g. `cd mli-0000-00-coursework`
- [ ] Check out the branch matching your GitHub handle:
  - e.g. `git checkout GITHUB_USERNAME`
  - All work must be committed in this branch, not "main".
- [ ] Quit any sub-shells using the `exit` command, if necessary.
  - Most terminal prompts put the sub-shell name at the start in parentheses.
- [ ] Initialize your Python environment and install dependencies:
  - `pipenv install`
- [ ] Link your workspace to the Pipenv environment:
  - Open the _coursework folder_ in your IDE
  - Using the IDE's explorer, open the first notebook file in the first lesson's folder.
  - Open your IDE's command palette (<kbd>Command + Shift + P</kbd> or equivalent).
  - Go to "Python: Select Interpreter".
  - Select the environment created in the previous step from the menu.
    - It will be named using the format _folder-name_hash_.
  - You can now click the "Select Kernel" option at the top of the notebook and choose the kernel corresponding the active environment.

Any notebooks and projects you open in this workspace should default to using the new coursework kernel, allowing you to execute Python cells in notebooks.

This single environment will be used to complete and run all exercises and walkthroughs for the course.

### Common Issues

#### "I don't see the interpreter in the list"

- Try refreshing the list using the reload icon at the top of the palette.
- OR manually enter the path to the interpreter:
  - Copy the path of the virtual environment from the command-line output of `pipenv install` or access it by running `pipenv --venv`
  - Paste it into the option "Enter Interpreter Path" in the interpreter selection menu.
  - Add `/bin/python` to the end of the path and confirm.
- OR add your system's "venv" path to your IDE settings:
  - Copy the path of the virtual environment from the command-line output of `pipenv install` or access it by running `pipenv --venv`.
  - In your IDE, go to Settings > "Python: Venv Path".
  - Paste the path into setting's text field.
  - Restart / reload your IDE and try selecting the interpreter again.

#### "I need to install different dependencies"

- Use your terminal to install new dependencies for platform-specific work, explorations, or projects.
  - Navigate to the coursework repo.
  - Use `pipenv install <dependency-name>`.
  - Ensure the [Pipfile](./Pipfile) reflects the change.

## Submitting Your Work

**Reminder:** Do not work in or push to the main branch of the repo.

- [ ] Complete and run the checkpoint cells in the notebooks you wish to submit
- [ ] Add and commit your work
- [ ] Push your work _to your branch_ on the repo:
  - `git push origin GITHUB_USERNAME`.

## Fetching updates

Content will be released periodically during the course. Follow either of the following workflows to download new releases.

### Workflow 1: Simple Merge

Use this if you are not familiar with rebasing:

- [ ] Save and commit the work in your branch.
- [ ] Switch to the main branch: `git checkout main`.
- [ ] Pull new changes: `git pull origin main`.
- [ ] Switch back to the branch you are working in: `git checkout GITHUB_USERNAME`.
- [ ] Merge the new changes in: `git merge main`.
  - Clean up any conflicts and commit the merge if necessary.

### Workflow 2: Rebase

Use this if you're familiar with rebasing and force-pushing:

- [ ] Save and commit the work in your branch.
- [ ] Pull new changes and rebase your branch: `git pull --rebase origin main`.
  - Resolve any conflicts that arise.
  - Add the resolved files with `git add .`.
  - Continue the rebase with `git rebase --continue`.

If the rebase has changed your commit history (which it often will), you may need to force push with `git push origin GITHUB_USERNAME --force`.

## Running notebooks on other platforms

You may choose to occasionally run the curriculum notebooks on platforms other than your local machine, particularly if local resources are slow or limited.

> ⚠︎ Do _not_ publicly share curriculum content.
>
> - Make sure any work done on online platforms is private.
> - Delete notebooks from online platforms once execution is complete and outputs have been saved and committed locally.

Notebook platforms include Google Colab and Kaggle. Most of the platforms do not support Python virtual environments.

To run notebooks on these platforms:

- [ ] Use the Import features of each platform to upload the notebook.
- [ ] Copy any supporting files, if necessary.
- [ ] In the terminal of the platform, install dependencies using `pip` based on the imports in the file.
  - Refer to the [Pipfile](./Pipfile) for specific versions.
