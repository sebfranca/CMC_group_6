"""Clone all FARMS repos"""

import os
import sys
from subprocess import check_call
try:
    from git import Repo
except ImportError:
    check_call([sys.executable, '-m', 'pip', 'install', 'GitPython'])
    from git import Repo


def install_dm_control_for_python_3_10():
    """Install dm_control for Python 3.10"""
    package = 'dm_control'
    print(f'Setting up {package}')
    if not os.path.isdir(package):
        repo = Repo.clone_from(
            'https://github.com/deepmind/dm_control.git',
            package,
        )
    repo = Repo(package)
    print(f'Active branch of {package}: {repo.active_branch.name}')
    print(f'Pulling latest version of {package}')
    repo.remotes.origin.pull()

    # Modify setup.py
    filename = f'{package}/setup.py'
    with open(filename, 'r', encoding='utf-8') as file_handler:
        content = file_handler.read()
    content = content.replace(
        '    python_requires=\'>=3.7, <=3.10\',',
        '    python_requires=\'>=3.7\',  # , <=3.10',
    )
    content = content.replace(
        '        \'labmaze\',',
        '        # \'labmaze\',',
    )
    with open(filename, 'w', encoding='utf-8') as file_handler:
        file_handler.write(content)

    # Modify requirements.txt
    filename = f'{package}/requirements.txt'
    with open(filename, 'r', encoding='utf-8') as file_handler:
        content = file_handler.read()
    content = content.replace('labmaze==1.0.3', '# labmaze==1.0.3')
    with open(filename, 'w', encoding='utf-8') as file_handler:
        file_handler.write(content)

    # print(f'Installing {package} dependencies')
    # check_call(pip_install + ['-r', 'requirements.txt'], cwd=package)

    print(f'Installing {package}')
    check_call([sys.executable, 'setup.py', 'install'], cwd=package)
    # check_call(pip_install + ['.', '-vvv'], cwd=package)


def main():
    """Main"""
    # pip_install = [sys.executable, '-m', 'pip', 'install']
    pip_install = ['pip', 'install']

    # Install MuJoCo
    check_call(pip_install + ['mujoco'])

    # Install dm_control for Python 3.10
    if sys.version_info >= (3, 10, 0):
        install_dm_control_for_python_3_10()
    else:  # Install dm_control for any other Python version
        check_call(pip_install + ['dm_control'])

    # FARMS
    for package in ['farms_core', 'farms_mujoco', 'farms_sim']:
        print(f'Providing option to reinstall {package} if already installed')
        check_call(['pip', 'uninstall', package])
        print(f'Installing {package}')
        check_call(
            pip_install
            + [
                '--no-cache-dir',
                f'https://gitlab.com/farmsim/{package}/'
                f'-/archive/cmc_2022/{package}-cmc_2022.zip',
                '-vvv',
            ]
        )
        print(f'Completed installation of {package}\n')


if __name__ == '__main__':
    main()
