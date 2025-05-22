# Create a simple HTML to compare the videos of different methods.

import os
import glob

experiments = {
    "plucker": "./assets/results_tmp/nv-resub-100k-re10k-ff-plucker-none/tests_zoom3x/",
    # "shared-gta": "./assets/results_tmp/nv-resub-100k-re10k-ff-shared-gta/tests",
    "plucker-pgl4-kse3-rope-invar-normK": "./assets/results_tmp/nv-resub-100k-re10k-ff-plucker-pgl4-kse3-rope-invar-normK/tests_zoom3x/",
}

output_dir = "comparision.html"

video_names = sorted(os.listdir(experiments["plucker"]))[10:100]

# Use table. The first row is the header with the name of the methods.
# The rest of the rows are the videos.
with open(output_dir, "w") as f:
    f.write("""
    <html>
    <body>
    """)

    # Write the header
    f.write("    <table>")
    f.write("        <tr>")
    f.write(f"            <th>name</th>")
    for method in experiments.keys():
        f.write(f"            <th>{method}</th>")
    f.write("        </tr>")

    # Write the videos
    for video_name in video_names:
        f.write("        <tr>")
        f.write(f"            <td>{video_name}</td>")
        for method in experiments:
            # <video width="100%" autoplay loop muted controls>
            #     <source
            #     src="./assets/results_tmp/nv-resub-100k-re10k-ff-plucker-pgl4-kse3-rope-invar-normK/tests/0b8e670f98cf5083.mp4"
            #     type="video/mp4" />
            # </video>
            f.write(f"            <td><video width='100%' autoplay loop muted controls>")
            f.write(f"                <source src='{experiments[method]}/{video_name}' type='video/mp4' />")
            f.write(f"            </video></td>")
        f.write("        </tr>")

    f.write("    </table>")
    f.write("</body>")
    f.write("</html>")
