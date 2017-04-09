"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:license: MIT

Testing suite for ML-Ensemble build.
"""

if __name__ == '__main__':
    import subprocess, os, sys, sysconfig

    # Check that nosetests exists
    print("Setting up tests...", end=" ", flush=True)

    has_nosetests = subprocess.run(["which", "nosetests"],
                                   stdout=open(os.devnull, "wb"),
                                   stderr=open(os.devnull, "wb"))

    # If not, try to install
    if has_nosetests.returncode:
        print("Could not find nosetests. Installing...", end=" ", flush=True)

        installation = subprocess.run(["pip", "install", "nose-exclude"],
                                      stdout=open(os.devnull, "wb"),
                                      stderr=open(os.devnull, "wb"))

        if installation.returncode:
            print("Installation successful.", end=" ", flush=True)
        else:
            print("Installation failed. Aborting test. "
                  "Ensure a valid version of "
                  "nosetests is installed (i.e. pip install "
                  "nose nose-exclude).")
            exit()

    print("Ready.", flush=True)

    # Run tests
    print("Checking build...", end=" ", flush=True)

    nosetests = subprocess.run(["nosetests", "-s", "-v", "--with-coverage"],
                               stdout=open(os.devnull, "wb"),
                               stderr=subprocess.PIPE)

    if nosetests.returncode == 0:
        print("Build ok.")
    else:
        print("Build failed.")
        print("Error log written to 'check_build_log.txt'.")

    with open("check_build_log.txt", "wb") as f:

        header = "-" * 22 + " Error log for testing mlens build " + "-" * 22
        build_start = "-" * 34 + " Build log " + "-" * 34
        python_version = "Python build: " + sys.version
        os_version = "OS platform: " + sysconfig.get_platform()

        try:
            import mlens
            mlens_version = "mlens version: " + mlens.__version__
        except Exception as e:
            mlens_version = "Cannot import mlens. Details:\n%r" % e

        for m in [header, python_version, os_version, mlens_version,
                  build_start]:
            f.write(bytes(m + "\n\n", "utf-8"))

        f.write(nosetests.stderr)
