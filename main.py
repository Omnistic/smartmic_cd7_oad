import socket

from bioio import BioImage

import numpy as np
import random
from cellpose import models
from scipy import ndimage
from skimage.measure import regionprops

import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots

import xml.etree.ElementTree as ET

from dotenv import load_dotenv
import os

PIXEL_SIZE_UM = 0.156
DETAILED_NUCLEI_PER_WELL = 5

load_dotenv()

overview_experiment_macro = os.getenv("overview_experiment_macro")
detail_experiment_macro = os.getenv("detail_experiment_macro")

# sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# sock.connect((os.getenv("cd_7_ip"), int(os.getenv("cd_7_port"))))
# sock.recv(4096)
# sock.sendall(f"RUN {overview_experiment_macro}\r\n".encode("ascii"))
# buf = b""
# while b"Ok" not in buf:
#     buf += sock.recv(4096)

image_file_path = os.getenv("image_file_path")
if not os.path.isfile(image_file_path):
    raise FileNotFoundError(f"image_file_path does not exist: {image_file_path}")

image = BioImage(
    image_file_path,
    include_subblock_metadata=True,
    use_aicspylibczi=True,
)

tile_regions = image.metadata.findall(".//TileRegion")

detail_experiment_path = os.getenv("detail_experiment_path")
exp_tree = ET.parse(detail_experiment_path)
base, ext = os.path.splitext(detail_experiment_path)
output_path = f"{base}_edited{ext}"
exp_root = exp_tree.getroot()
regions_container = exp_root.find(".//SingleTileRegions")
existing = regions_container.findall("SingleTileRegion")
template_region = existing[0]
z_value = template_region.findtext("Z")

for region in existing:
    regions_container.remove(region)

model = models.CellposeModel(gpu=True)

flow_threshold = 0.4
cellprob_threshold = 0.0
tile_norm_blocksize = 0

def bin4x4(arr):
    h, w = arr.shape[:2]
    h4, w4 = h // 4, w // 4
    return arr[:h4 * 4, :w4 * 4].reshape(h4, 4, w4, 4).mean(axis=(1, 3))

def bin4x4_mask(arr):
    return arr[::4, ::4]

def discrete_mask_colorscale(n_labels):
    palette = pc.qualitative.Dark24
    colors = ['black'] + list((palette * ((n_labels // len(palette)) + 1))[:n_labels])
    n = len(colors)
    cs = []
    for i, c in enumerate(colors):
        cs += [[i / n, c], [(i + 1) / n - 1e-10, c]]
    cs[-1][0] = 1.0
    return cs

position_counter = 1
for tile_region in tile_regions:
    well_name = tile_region.get("Name")
    image.set_scene(f"{well_name}-{well_name}")
    region_center_str = tile_region.findtext("CenterPosition")
    rcx_um, rcy_um = map(float, region_center_str.split(","))
    dapi_index = next(i for i, name in enumerate(image.channel_names) if 'dapi' in str(name).lower())
    nuclei = np.amax(image.get_image_data("YXZ", C=dapi_index), axis=2)
    nuclei_masks, _, _ = model.eval(nuclei, batch_size=32, flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold, normalize={"tile_norm_blocksize": tile_norm_blocksize})

    number_of_nuclei = np.max(nuclei_masks)
    print(f"Found {number_of_nuclei} nuclei in {well_name}.")

    props = regionprops(nuclei_masks)

    solidities = np.array([pp.solidity for pp in props])
    solidity_min = np.median(solidities) - np.median(np.abs(solidities - np.median(solidities)))
    print(f"Calculated solidity threshold of {solidity_min:.3f} for valid nuclei in {well_name}.")
    valid = [pp for pp in props if pp.solidity >= solidity_min]

    areas = np.array([pp.area for pp in valid])
    area_median = np.median(areas)
    area_mad = np.median(np.abs(areas - area_median))
    valid = [pp for pp in valid if area_median - area_mad <= pp.area <= area_median + area_mad]
    print(f"Calculated area threshold of {area_median:.3f} ± {area_mad:.3f} for valid nuclei in {well_name}.")

    valid_indices = [pp.label for pp in valid]
    print(f"Found {len(valid)} valid nuclei in {well_name}.")

    selected_nuclei_indices = random.sample(valid_indices, min(DETAILED_NUCLEI_PER_WELL, len(valid_indices)))
    print(f"Selected nuclei with indices {selected_nuclei_indices} for detailed acquisition in {well_name}.")

    n = int(number_of_nuclei)
    cs = discrete_mask_colorscale(n)
    mask_kwargs = dict(colorscale=cs, zmin=0, zmax=n, showscale=False)
    fig = make_subplots(rows=2, cols=3, subplot_titles=["Nuclei", "Nuclei Masks", "Valid Nuclei Masks", "Invalid Nuclei Masks", "Selected Nuclei Masks"])
    fig.add_trace(go.Heatmap(z=bin4x4(nuclei), colorscale='gray', showscale=False), row=1, col=1)
    fig.add_trace(go.Heatmap(z=bin4x4_mask(nuclei_masks), **mask_kwargs), row=1, col=2)
    fig.add_trace(go.Heatmap(z=bin4x4_mask(np.where(np.isin(nuclei_masks, valid_indices), nuclei_masks, 0)), **mask_kwargs), row=1, col=3)
    fig.add_trace(go.Heatmap(z=bin4x4_mask(np.where((nuclei_masks > 0) & ~np.isin(nuclei_masks, valid_indices), nuclei_masks, 0)), **mask_kwargs), row=2, col=1)
    fig.add_trace(go.Heatmap(z=bin4x4_mask(np.where(np.isin(nuclei_masks, selected_nuclei_indices), nuclei_masks, 0)), **mask_kwargs), row=2, col=2)
    fig.update_layout(
        xaxis1=dict(scaleanchor="y1", scaleratio=1),
        xaxis2=dict(scaleanchor="y2", scaleratio=1),
        xaxis3=dict(scaleanchor="y3", scaleratio=1),
        xaxis4=dict(scaleanchor="y4", scaleratio=1),
        xaxis5=dict(scaleanchor="y5", scaleratio=1),
        yaxis1=dict(autorange="reversed"),
        yaxis2=dict(autorange="reversed"),
        yaxis3=dict(autorange="reversed"),
        yaxis4=dict(autorange="reversed"),
        yaxis5=dict(autorange="reversed"),
    )
    fig.write_html(f"{well_name}.html")

    for ii, nucleus_index in enumerate(selected_nuclei_indices):
        ncy_py, ncx_px = ndimage.center_of_mass(nuclei_masks == nucleus_index)
        rel_ncx_um = ncx_px * PIXEL_SIZE_UM
        rel_ncy_um = ncy_py * PIXEL_SIZE_UM
        abs_ncx_um = rcx_um - image.dims.X * PIXEL_SIZE_UM / 2 + ncx_px * PIXEL_SIZE_UM
        abs_ncy_um = rcy_um - image.dims.Y * PIXEL_SIZE_UM / 2 + ncy_py * PIXEL_SIZE_UM

        new_region = ET.SubElement(regions_container, "SingleTileRegion")
        new_region.set("Name", f"P{position_counter}")
        new_region.set("Id", str(random.randint(10**17, 10**18 - 1)))
        ET.SubElement(new_region, "X").text = f"{abs_ncx_um:.3f}"
        ET.SubElement(new_region, "Y").text = f"{abs_ncy_um:.3f}"
        ET.SubElement(new_region, "Z").text = z_value
        ET.SubElement(new_region, "IsUsedForAcquisition").text = "true"
        ET.SubElement(new_region, "AdditionalValues")
        position_counter += 1

with open(output_path, "wb") as f:
    f.write(b"\xef\xbb\xbf")
    exp_tree.write(f, encoding="utf-8", xml_declaration=True)

# sock.sendall(f"RUN {detail_experiment_macro}\r\n".encode("ascii"))
# buf = b""
# while b"Ok" not in buf:
#     buf += sock.recv(4096)