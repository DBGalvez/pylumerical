# # Simple far field projection example - Python style commands
#
# A simple example using PyLumerical to setup and run a simple far field projection example
#
# Sets up and runs a Far Field Projection FDTD simulation.
# Demonstrates initializing objects using keyword arguments and OrderedDict.

# ## Prerequisites:
#
# Valid FDTD license is required.
#
# ### Perform required imports

# +

from collections import OrderedDict
import math
from pathlib import Path

import numpy as np

import ansys.lumerical.core as lumapi

# -

# some definitions

RESOLUTION = 101
LINE_POINTS = 100

# helper function for calculating the far field


def calculate_exact_far_field(fdtd, monitor, x, y, z):
    """Calculate the exact far field projection."""
    result = np.zeros(len(x))

    for i in range(len(x)):
        e_field = fdtd.farfieldexact3d(monitor, x[i], y[i], z[i])
        result[i] = np.sum(np.abs(e_field) ** 2)
    return result


# main script ---------------------------------------------
# ### Open interactive session with the "with" context manager, run session, retrieve and plots results, and close session

# Set hide = True to hide the Lumerical GUI.
with lumapi.FDTD() as fdtd:
    # Set up simulation region using keyword arguments
    fdtd.addfdtd(x=0, x_span=8e-6, y=0, y_span=8e-6, z=0.25e-6, z_span=0.5e-6)

    # Set up source using Python OrderedDict
    # OrderedDict is recommended when order is important
    # Here, the scalar approximation prop should be set before waist radius
    props = OrderedDict(
        [
            ("injection axis", "z"),
            ("direction", "forward"),
            ("angle theta", 30),
            ("angle phi", 15),
            ("x", 0),
            ("x span", 16e-6),
            ("y", 0),
            ("y span", 16e-6),
            ("z", 0.2e-6),
            ("use scalar approximation", 1),
            ("waist radius w0", 2e-6),
            ("distance from waist", 0),
            ("wavelength start", 1e-6),
            ("wavelength stop", 1e-6),
        ]
    )
    fdtd.addgaussian(properties=props)

    # Set up monitors using regular dict
    props = {"monitor type": "2D Z-normal", "name": "z2", "x": 0, "x span": 14.2e-6, "y": 0, "y span": 11e-6, "z": 0.3e-6}
    fdtd.adddftmonitor(properties=props)

    props = {"monitor type": "Point", "name": "time_monitor", "x": 3, "y": 0, "z": 0.3e-6}
    fdtd.addtime(properties=props)

    # Run and save simulation

    save_file = Path(__file__).parent / "solver_far_field_projection_tutorial.fsp"
    fdtd.save(str(save_file))
    fdtd.run()

    # Retrieve and plot results
    E2 = fdtd.farfield3d("z2")
    ux = fdtd.farfieldux("z2")
    uy = fdtd.farfielduy("z2")

    fdtd.image(ux, uy, E2, "", "", "|E|^2 at 1 m", "polar")

    input("Press Enter to continue...")
    #########################################################################################################################
    # 	                                 Integrate power in the far field,
    #                                      normalized to the source power

    print("Integrate power in the far field, normalized to the source power ------------------")

    ## choose the half angle over which we will integrate
    half_angle = 30  # in degrees

    # choose the central angle of the cone
    cone_center_theta = 0
    cone_center_phi = 0

    # Display the half angle in the terminal
    print(f"     The half angle is: {half_angle} degrees at (theta,phi)=({cone_center_theta}, {cone_center_phi})")

    # collect the far field data
    E2 = fdtd.farfield3d("z2")  # this returns |E|^2 in the far field
    ux = fdtd.farfieldux("z2")
    uy = fdtd.farfielduy("z2")

    ################ Method1
    temp_cone = fdtd.farfield3dintegrate(E2, ux, uy, half_angle, cone_center_theta, cone_center_phi)
    temp_far = fdtd.farfield3dintegrate(E2, ux, uy)

    # get fraction of power in cone, then normalize with monitor transmission
    T = temp_cone / temp_far * fdtd.transmission("z2")
    print(f"     The normalized transmission by method 1 is: {T * 100} %")

    ################ Method2
    # calculate and integrate the Poynting vector
    eps0 = 8.85419e-12
    mu0 = 1.25664e-06
    temp2 = 0.5 * math.sqrt(eps0 / mu0) * fdtd.farfield3dintegrate(E2, ux, uy, half_angle, cone_center_theta, cone_center_phi)
    # apply source normalization
    T = temp2 / fdtd.sourcepower(fdtd.getdata("z2", "f"))

    print(f"     The normalized transmission by method 2 is: {T * 100} %")

    input("Press Enter to continue...")

    ##########################################################
    # 	Calculate the complex electric field components in
    # the far field, in cartesian and spherical coordinate systems

    print("Calculate the complex electric field components in the far field, in cartesian, and spherical coordinate systems--------")

    # project in cartesian coordinate system
    E = fdtd.farfieldvector3d("z2")
    Ex = fdtd.pinch(E, 3, 1)
    Ey = fdtd.pinch(E, 3, 2)
    Ez = fdtd.pinch(E, 3, 3)
    ux = fdtd.farfieldux("z2")
    uy = fdtd.farfielduy("z2")
    fdtd.image(ux, uy, np.abs(Ex), "", "", "|Ex|", "polar")
    fdtd.image(ux, uy, np.abs(Ey), "", "", "|Ey|", "polar")
    fdtd.image(ux, uy, np.abs(Ez), "", "", "|Ez|", "polar")

    # project in spherical (polar) coordinate system
    E = fdtd.farfieldpolar3d("z2", 1, 201, 201)
    Er = fdtd.pinch(E, 3, 1)
    Etheta = fdtd.pinch(E, 3, 2)
    Ephi = fdtd.pinch(E, 3, 3)
    ux = fdtd.farfieldux("z2", 1, 201, 201)
    uy = fdtd.farfielduy("z2", 1, 201, 201)
    fdtd.image(ux, uy, np.abs(Er), "", "", "|Er|", "polar")
    fdtd.image(ux, uy, np.abs(Etheta), "", "", "|Etheta|", "polar")
    fdtd.image(ux, uy, np.abs(Ephi), "", "", "|Ephi|", "polar")

    input("Press Enter to continue...")

    ##########################################################
    ## 	Create line plots of far field data.
    print("Create line plots of far field data")
    ## pause before running next portion of script

    m = "z2"
    RESOLUTION = 201
    E2 = fdtd.farfield3d(m, 1, RESOLUTION, RESOLUTION)
    ux = fdtd.farfieldux(m, 1, RESOLUTION, RESOLUTION)
    uy = fdtd.farfielduy(m, 1, RESOLUTION, RESOLUTION)

    # Figure: standard image plot
    fdtd.image(ux, uy, E2, "", "", "|E|^2", "polar")

    # Figure: line plot of E2 vs angle at uy=0 (phi=0).
    theta = np.linspace(-90, 90, LINE_POINTS)
    phi = 0
    fdtd.plot(theta, fdtd.farfieldspherical(E2, ux, uy, theta, phi), "Theta", "E^2 far", "Phi=" + str(phi))

    # Figure: plot E2 vs theta at phi=15
    # define desired plot
    phi = 15
    theta = np.linspace(-90, 90, LINE_POINTS * 2)
    fdtd.plot(theta, fdtd.farfieldspherical(E2, ux, uy, theta, phi), "theta", "E^2 far", "Phi=" + str(phi))

    # Figure: plot E2 vs phi at theta=30
    phi = np.linspace(-180, 180, LINE_POINTS * 2)
    theta = 30
    fdtd.plot(phi, fdtd.farfieldspherical(E2, ux, uy, theta, phi), "Phi", "E^2 far", "theta=" + str(theta))

    ############################################
    ## Calculate the far field distribution at 1m and 10um with the farfield3d and
    ## farfieldexact3d functions.  These functions will return the same result in the far field.
    ## In the intermediate field, only the farfieldexact3d function is valid.

    ## pause before running next portion of script
    input("Press Enter to continue...")

    ## set monitor name
    m = "z2"

    ## calculate field distribution at 1m.------------------------

    ## calculate various position vectors

    ux = fdtd.farfieldux(m, 1, RESOLUTION, 1)
    uy = fdtd.farfielduy(m, 1, RESOLUTION, 1)
    uz = np.sqrt(1 - (ux**2) - (uy**2))
    r = 1
    theta = np.acos(uz) * np.sign(ux) * 180 / np.pi
    if round(RESOLUTION / 2) != RESOLUTION / 2:  # RESOLUTION=odd number, avoid np.abs(ux)/ux
        theta[round(RESOLUTION / 2)] = 0  # to make sure theta at centre is zero
    x = r * np.sin(theta * np.pi / 180)
    y = x * 0
    z = r * np.cos(theta * np.pi / 180)

    ## Standard far field projection function
    E2far1 = fdtd.farfield3d(m, 1, RESOLUTION, 1)

    ## Far field exact function
    E2exact1 = calculate_exact_far_field(fdtd, m, x, y, z)

    # plot results
    fdtd.plot(theta, E2far1, E2exact1, "theta", "|E|^2", "1m radius")
    fdtd.legend("farfield3d", "farfieldexact3d")

    ## calculate at distance of 10um----------------------------
    r = 10e-6
    x = r * np.sin(theta * np.pi / 180)
    y = x * 0
    z = r * np.cos(theta * np.pi / 180)

    ## Standard far field projection function
    ## There is no need to recalculate the projection.
    ## Simply scale the fields by 1/r^2
    E2far2 = E2far1 / r**2

    ## Far field exact function
    E2exact2 = calculate_exact_far_field(fdtd, m, x, y, z)

    fdtd.plot(theta, E2far2, E2exact2, "theta", "|E|^2", "10um radius")
    fdtd.legend("farfield3d scaled", "farfieldexact3d")

    ## calculate at a flat plane at x=-10:10,y=0,z=10
    x = np.linspace(-20e-6, 20e-6, LINE_POINTS + 1)
    y = 0
    z = 10e-6
    E2exact3 = fdtd.farfieldexact3d(m, x, y, z)
    E2exact3 = np.abs(E2exact3[0:101, 0, 0, 0]) ** 2 + np.abs(E2exact3[0:101, 0, 0, 1]) ** 2 + np.abs(E2exact3[0:101, 0, 0, 2]) ** 2
    fdtd.plot(x * 1e6, E2exact3, "x (um)", "|E|^2", "line at y=0, z=10um")
    fdtd.legend("farfieldexact3d")

    print("Example complete. Press Enter to close.")
    input()
