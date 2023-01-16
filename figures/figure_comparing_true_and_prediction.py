from PIL import Image
import pandas as pd
import texfig  # import texfig first to configure Matplotlib's backend

SELECTED_INDICES = [500, 700, 1000, 1200]


def main():

    dataframe_original = pd.read_csv("dataset.csv")
    dataframe_unet = pd.read_csv("dataset_unet.csv")

    ratio = 1.0

    fig, ax = texfig.subplots(ratio=ratio, nrows=4, ncols=4)

    for i, index in enumerate(SELECTED_INDICES):
        path = dataframe_original["images_vel_x"][index]
        image_vel_x = Image.open(path)

        path = dataframe_original["images_vel_y"][index]
        image_vel_y = Image.open(path)

        path = dataframe_unet["images_unet"][index]
        image_pressure_unet = Image.open(path)

        path = dataframe_original["images_pressure"][index]
        image_pressure_original = Image.open(path)

        ax[0, i].imshow(image_vel_x)
        ax[1, i].imshow(image_vel_y)
        ax[2, i].imshow(image_pressure_unet)
        ax[3, i].imshow(image_pressure_original)
        ax[0, i].axis("off")
        ax[1, i].axis("off")
        ax[2, i].axis("off")
        ax[3, i].axis("off")

        ax[0, 0].text(
            0.06,
            0.35,
            r"$u$-velocity",
            rotation="vertical",
            transform=ax[0, 0].transAxes,
            ha="center",
            size=10,
        )
        ax[1, 0].text(
            0.06,
            0.35,
            r"$v$-velocity",
            rotation="vertical",
            transform=ax[1, 0].transAxes,
            ha="center",
            size=10,
        )
        ax[2, 0].text(
            0.06,
            0.33,
            r"Predicted $Cp$",
            rotation="vertical",
            transform=ax[2, 0].transAxes,
            ha="center",
            size=10,
        )
        ax[3, 0].text(
            0.06,
            0.25,
            r"True $Cp$",
            rotation="vertical",
            transform=ax[3, 0].transAxes,
            ha="center",
            size=10,
        )

    fig.subplots_adjust(
        left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0.02, wspace=0.02
    )

    texfig.savefig(
        "Fig_Cp_field_prediction", dpi=1000, bbox_inches="tight", pad_inches=0
    )


if __name__ == "__main__":
    main()
