{
  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    nixrik = {
      url = "gitlab:erikmannerfelt/nixrik";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {self, nixpkgs, nixrik}: {
    devShells = nixrik.extra.lib.for_all_systems(pkgs_pre: (
      let
        pkgs = pkgs_pre.extend nixrik.overlays.python_extra;
        my-python = pkgs.python311PackagesExtra.from_requirements ./requirements.txt;
      in {
        default = pkgs.mkShell {
          name = "IncoherentSurges";
          buildInputs = with pkgs; [
            my-python
          ];
        };
      }
    ));
  };
}
