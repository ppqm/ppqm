# physical constants and unit convertion

accepted_units = [
    "ev",
    "kcalmol",
]

calories_to_joule = 4.184
hartree_to_ev = 27.211386245988
hartree_to_kcalmol = 627.5094740631
ev_to_kcalmol = hartree_to_kcalmol / hartree_to_ev
kcalmol_to_ev = hartree_to_ev / hartree_to_kcalmol


# Physical constants
k_kcalmolkelvin = 0.001985875  # kcal/(mol *k)
kelvin_room = 298.15  # kelvin
