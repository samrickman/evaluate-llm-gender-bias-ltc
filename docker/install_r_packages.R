install.packages("jsonlite")
install.packages("remotes")

pkgs <- jsonlite::read_json("./docker/r_packages.json", simplifyVector = FALSE)

for (pkg in pkgs) {
    message("Installing: ", pkg)
    remotes::install_version(
        pkg$Package,
        pkg$Version
    )
}
