#!/bin/bash

# =============================================================================
# HELPER ACTIONS
# =============================================================================

NC=$(echo "\033[m")
BOLD=$(echo "\033[1;39m")
CMD=$(echo "\033[1;34m")
OPT=$(echo "\033[0;34m")

action_usage(){
    echo -e "  _   _                              _                   _                  "
    echo -e " | | | |_   _ _ __   __ _  __ _ _ __(_) __ _ _ __       | |    ___  ___ ___ "
    echo -e " | |_| | | | | '_ \\ / _\` |/ _\` | '__| |/ _\` | '_ \\ _____| |   / _ \\/ __/ __|"
    echo -e " |  _  | |_| | | | | (_| | (_| | |  | | (_| | | | |_____| |__| (_) \\__ \\__ \\"
    echo -e " |_| |_|\\__,_|_| |_|\\__, |\\__,_|_|  |_|\\__,_|_| |_|     |_____\\___/|___/___/"
    echo -e "                    |___/ The loss function based on the Hungarian algorithm"
    echo -e ""
    echo -e "${BOLD}System Commands:${NC}"
    echo -e "   ${CMD}init${NC}  initializers environment;"
    echo -e "   ${CMD}test${OPT} ...${NC} runs tests;"
    echo -e "      ${OPT}-c ${NC}generates code coverage summary;"
    echo -e "      ${OPT}-r ${NC}generates code coverage report;"
    echo -e "   ${CMD}prep${NC}  makes pre-commit formatting and checking;"
    echo -e "   ${CMD}build${NC} generates distribution archives;"
}

action_init(){
    if [ -d .venv ];
        then
            rm -r .venv
    fi

    python3 -m venv .venv
    source .venv/bin/activate
    pip3 install -r requirements.txt --no-cache
    pre-commit autoupdate
}

action_run(){
    source .venv/bin/activate
    python main.py
}

action_test(){
    source .venv/bin/activate

    OPTS=()
    while getopts ":m:cr" opt; do
        case $opt in
            c)
                OPTS+=(--cov=hungarian_loss)
                ;;
            r)
                OPTS+=(--cov-report=xml:cov.xml)
                ;;
            \?)
                echo -e "Invalid option: -$OPTARG"
                exit
                ;;
        esac
    done

    pytest --capture=no -p no:warnings ${OPTS[@]}
}

action_prep(){
    source .venv/bin/activate
    pre-commit run --all-files
}

action_build(){
    source .venv/bin/activate
    python -m build
}

# =============================================================================
# HELPER COMMANDS SELECTOR
# =============================================================================
case $1 in
    init)
        action_init
    ;;
    run)
        action_run
    ;;
    test)
        action_test ${@:2}
    ;;
    prep)
        action_prep ${@:2}
    ;;
    build)
        action_build
    ;;
    *)
        action_usage
    ;;
esac

exit 0
