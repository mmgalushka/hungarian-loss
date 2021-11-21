# How to contribute to Hungarian Loss

Thank you for considering contributing to Hungarian Loss project!

## Get Started

### Configure Git & Clone Repository

* Download and install the [latest version of git](https://git-scm.com/downloads).

* Configure git by setting up your username and email.

```bash
~$ git config --global user.name 'your name'
~$ git config --global user.email 'your email'
```

* Create your [GitHub account](https://github.com/join) if you don't have one already.

* Fork Hungarian Loss to your GitHub account using the Fork button.

* [Clone](https://docs.github.com/en/github/getting-started-with-github/fork-a-repo#step-2-create-a-local-clone-of-your-fork) the main repository locally.

```bash
~$ git clone https://github.com/mmgalushka/hungarian-loss.git
~$ cd hungarian-loss
```

* Add fork as a remote where you will push your work to. Replace `{username}` with your username.

```bash
~$ git remote add fork https://github.com/{username}/hungarian-loss
```

### Initialize Environment

* Use `helper.sh` to create a virtual environment and initialize required dependencies.

```bash
~$ ./helper.sh init
```

**Note:**  `helper.sh` can be viewed as your "command centre". It allows performer various useful operations with the repository. Read this section to learn more about how to use the helper.

* Run test to make sure that all work well

```bash
~$ ./helper.sh test
```

It should be no error :wink:.

### Start Coding

* Create a branch to identify the issue you would like to work on. It is advisable to use the following convention for naming your issue: `issue-{number}`, where `number` is an issue identifier ex. `issue-123`.

```bash
~$ git fetch origin
~$ git checkout -b your-branch-name origin/main
```

* Using your favorite editor to start coding. [Commit](https://dont-be-afraid-to-commit.readthedocs.io/en/latest/git/commandlinegit.html#commit-your-changes) changes regularly.

* Make sure you have all tests that cover any code changes you make. This project requires 100% coverage, no exception!

```bash
~$ ./helper.sh test
```

* Push your commits to your fork on GitHub and create a [pull request](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request). Link to the issue being addressed with `fixes #123` in the pull request.

```bash
~$ git push --set-upstream fork your-branch-name
```

## Helper Usage

`helper.sh` provides a command-line shell to automate a lot of routine operations. To see the available commands call `helper.sh` without options.

```text
~$ ./helper.sh

  _   _                              _                   _
 | | | |_   _ _ __   __ _  __ _ _ __(_) __ _ _ __       | |    ___  ___ ___
 | |_| | | | | '_ \ / _` |/ _` | '__| |/ _` | '_ \ _____| |   / _ \/ __/ __|
 |  _  | |_| | | | | (_| | (_| | |  | | (_| | | | |_____| |__| (_) \__ \__ \
 |_| |_|\__,_|_| |_|\__, |\__,_|_|  |_|\__,_|_| |_|     |_____\___/|___/___/
                    |___/ The loss function based on the Hungarian algorithm

System Commands:
   init  initializers environment;
   test ... runs tests;
      -c generates code coverage summary;
      -r generates code coverage report;
   prep  makes pre-commit formatting and checking;
   build generates distribution archives;
```

### Init

`init` command performs initialization of Hungarian Loss project from scratch. Usually, it should be called just once  (straight after cloning repository).  First, it creates a virtual environment and installs necessary dependencies defined in the `requirements.txt` file. If you introduce a new dependency (to `requirements.txt` file) run the `init` command again.

### Test

`test` command helps to test Hungarian Loss project using `pytest`. It can be run with the following option.

| Option      | Description |
|:-----------:|:------------|
| -c          | generates code coverage summary (% of tested code) |
| -r          | generates code coverage report (to see what is covered by tests and what is not) |

More information about code coverage can be found [here](https://pytest-cov.readthedocs.io/en/latest/).

---
NOTE

This project requires 100% coverage! Please make sure  you introduced all necessary tests, before launching a pull request. Just run test command with coverage option `-c`.

```bash
~$ ./helper.sh test -c
```

If you are not sure which parts of your code are covered by tests and which do not use the following command to generate a test coverage report.

```bash
~$ ./helper.sh test -cr
```

After executing this command you can use for example [Coverage Gutters extension](https://marketplace.visualstudio.com/items?itemName=ryanluker.vscode-coverage-gutters) to inspect the coverage report.

---

### Preparation

Use the `prep` command to perform Black formatting and check the code for compliance with Flake8. This step is not necessary since this project has pre-commit hooks for  Black formatting and Flake8 checks. However, by executing this command you will save yourself a hassle in recommitting code again after activating the hooks.

```bash
~$ ./helper.sh prep
```

### Build

Run the `build` command for creating the Hungarian Loss wheel. This command should be useful if you make some changes to the project code and would like to test PIP installation locally.

```bash
~$ ./helper.sh build
```
