{
  description = "Python environment with Nix-managed packages";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          pandas
          scipy
          numpy
          pip
          langchain
          langchain-community
          langchain-ollama
          chromadb
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pythonEnv
            pkgs.stdenv.cc.cc.lib
          ];

          shellHook = ''
            # Create a symlink that PyCharm can use as interpreter
            mkdir -p .nix-python
            ln -sf ${pythonEnv}/bin/python .nix-python/python
            echo "Python interpreter available at: $PWD/.nix-python/python"
            echo "Use this path in PyCharm's Python Interpreter settings"
          '';
        };
      });
}