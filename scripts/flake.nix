{
  description = "Python environment with local pip";

  inputs = {
    nixpkgs.url = "/home/volhovm/code/nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        mypython = (pkgs.python3.withPackages (python-packages: with python-packages; [
              pandas
              scipy
              numpy
              pip
              virtualenv
              langchain
              langchain-community
              langchain-ollama
              chromadb
            ]));
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            #mypython
            #pip
            pkgs.python3Packages.pip
            pkgs.python3
            pkgs.stdenv.cc.cc.lib  # Add this to provide libstdc++
            #python.pkgs.pip
            #python.pkgs.virtualenv
            #pkgs.rlama  # Use pkgs.rlama instead of just rlama
            #python.pkgs.langchain
            #python.pkgs.langchain-community
            #python.pkgs.langchain-ollama
            #python.pkgs.chromadb
          ];

          shellHook = ''
            # Create a virtual environment if it doesn't exist
            if [ ! -d .venv ]; then
              ${pkgs.python3}/bin/python -m venv .venv
            fi

            # Activate the virtual environment
            source .venv/bin/activate

            # Clear PYTHONPATH to avoid import conflicts
            unset PYTHONPATH

            # Add only the site-packages directory to PYTHONPATH
            export PYTHONPATH="$PWD/.venv/lib/python3.13/site-packages:$PYTHONPATH"
            export PATH="$PWD/.venv/bin:$PATH"


            export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH

            echo "Python virtual environment activated. Use 'pip' to install packages locally."
          '';
        };
      });
}
