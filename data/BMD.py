import cv2
import decord
import glob
import numpy as np
import os
import pickle
import re
import subprocess
import torch
import zipfile

from decord import VideoReader, cpu
from PIL import Image
from sklearn.datasets import get_data_home
from tqdm import tqdm
from typing import Optional, List, Dict, Tuple, Union, Callable

from data.base import BaseDataset  # Make sure this import works correctly

# Noise ceiling threshold for filtering voxels
NCSNR_THRESHOLD = 0


class BMDStimulusSet(BaseDataset):
    """Dataset for the NSD stimulus set (images)."""

    def __init__(
        self,
        root_dir: Optional[str] = None,
        overwrite: bool = False,
        preprocess: Optional[Callable] = None,
        train: bool = True
    ):
        """
        Initialize the NSDStimulusSet.

        Args:
            root_dir: Root directory.
            overwrite: Overwrite existing files.
            preprocess: Preprocessing transform.
        """
        super().__init__(root_dir)
        self.overwrite = overwrite
        self.stimulus_data = None
        self.train = train
        self.preprocess = preprocess
        self.fps = 10.0

    def _load_data(self, video_path: str) -> List[Image.Image]:
        """Load video via Decord.get_batch, sample at self.fps, and return exactly 30 frames."""
        # Open with Decord on CPU
        vr = VideoReader(video_path, ctx=cpu(0))

        # Fetch FPS (fallback to 30 if missing)
        orig_fps = vr.get_avg_fps() or 30.0
        interval = max(int(round(orig_fps / self.fps)), 1)

        # Build and truncate to the first 30 sample‐indices
        idxs = list(range(0, len(vr), interval))[:30]

        # Pull them all at once as a tensor of shape [N, H, W, 3]
        tensor_batch = vr.get_batch(idxs)

        # Convert each frame to a PIL Image
        frames = [Image.fromarray(frame.numpy()) for frame in tensor_batch]

        # If fewer than 30, repeat the last frame
        if frames:
            while len(frames) < 30:
                frames.append(frames[-1])
        else:
            raise ValueError("No frames were loaded from the video.")

        return frames

    def _download_bmd_data(self):
        url = f'https://boldmomentsdataset.csail.mit.edu/stimuli_metadata/stimulus_set.zip'

        if not os.path.exists(self.root_dir) or self.overwrite:
            self.fetch(
                source=url,
                filename=os.path.basename(url),
                force_download=self.overwrite,
            )

            filepath = os.path.join(self.root_dir, os.path.basename(url))
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(path=self.root_dir, pwd=b'goboldmoments!')

    def _prepare_videos(self):
        """Load and preprocess the test images."""
        self._download_bmd_data()
        if self.train:
            self.stimulus_data = [
                os.path.join(self.root_dir, 'stimulus_set',
                             'mp4_h264', f"{n:04d}.mp4")
                for n in range(1, 1001)
            ]
        else:
            self.stimulus_data = [
                os.path.join(self.root_dir, 'stimulus_set',
                             'mp4_h264', f"{n:04d}.mp4")
                for n in range(1001, 1103)
            ]

    def __len__(self):
        """Return the number of test images."""
        if self.stimulus_data is None:
            self._prepare_videos()
        return len(self.stimulus_data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return a preprocessed image."""
        if self.stimulus_data is None:
            self._prepare_videos()
        video = self.stimulus_data[idx]
        return self.preprocess(self._load_data(video))


class BMDStimulusTrainSet(BMDStimulusSet):
    """Dataset class for the training set of the BMD Stimulus."""

    def __init__(
        self,
        root_dir: Optional[str] = None,
        overwrite: bool = False,
        preprocess: Optional[Callable] = None
    ):
        # Initialize with train=True to load the training videos
        root = os.path.join(
            get_data_home(), BMDStimulusSet.__name__
        )
        super().__init__(root_dir=root, overwrite=overwrite, preprocess=preprocess, train=True)


class BMDStimulusTestSet(BMDStimulusSet):
    """Dataset class for the testing set of the BMD Stimulus."""

    def __init__(
        self,
        root_dir: Optional[str] = None,
        overwrite: bool = False,
        preprocess: Optional[Callable] = None
    ):
        # Initialize with train=False to load the test videos
        root = os.path.join(
            get_data_home(), BMDStimulusSet.__name__
        )
        super().__init__(root_dir=root, overwrite=overwrite,
                         preprocess=preprocess, train=False)


class BMDAssembly(BaseDataset):
    """
    Dataset class for the OpenNeuro ds005165 fMRI data (assembly).
    # annotations.json (not downloaded here) give video metadata
    """

    def __init__(
        self,
        root_dir: Optional[str] = None,
        subjects: Optional[List[int]] = None,
        rois: Optional[List[str]] = None,  # Keep for consistency
        overwrite: bool = False,
        ncsnr_threshold: float = NCSNR_THRESHOLD
    ):
        """
        Initialize OpenNeuroAssembly.

        Args:
            root_dir: Root directory.
            subjects: List of subject IDs or None for all.
            rois: List of ROIs (for consistency, but get_assembly combines).
            overwrite: Overwrite existing files.
        """
        super().__init__(root_dir)
        self.url = (
            "s3://openneuro.org/ds005165/derivatives/"
            "versionB/MNI152/GLM/"
        )
        self._available_subjects = list(range(1, 11))
        self._available_rois: Optional[List[str]] = None  # Lazy load
        # Define regions based on ROI prefixes/suffixes
        self.regions_map = {
            '7AL': ['l7AL', 'r7AL'],
            'BA2': ['lBA2', 'rBA2'],
            'EBA': ['lEBA', 'rEBA'],
            'FFA': ['lFFA', 'rFFA'],
            'IPS0': ['lIPS0', 'rIPS0'],
            'IPS1-2-3': ['lIPS1-2-3', 'rIPS1-2-3'],
            'LOC': ['lLOC', 'rLOC'],
            'MT': ['lMT', 'rMT'],
            'OFA': ['lOFA', 'rOFA'],
            'PFop': ['lPFop', 'rPFop'],
            'PFt': ['lPFt', 'rPFt'],
            'PPA': ['lPPA', 'rPPA'],
            'RSC': ['lRSC', 'rRSC'],
            'STS': ['lSTS', 'rSTS'],
            'TOS': ['lTOS', 'rTOS'],
            'V1d': ['lV1d', 'rV1d'],
            'V1v': ['lV1v', 'rV1v'],  # Left and right V1v together
            'V1': ['lV1d', 'rV1d', 'lV1v', 'rV1v'],
            'V2d': ['lV2d', 'rV2d'],
            'V2v': ['lV2v', 'rV2v'],
            'V2': ['lV2d', 'rV2d', 'lV2v', 'rV2v'],
            'V3ab': ['lV3ab', 'rV3ab'],
            'V3d': ['lV3d', 'rV3d'],
            'V3v': ['lV3v', 'rV3v'],
            'V3': ['lV3v', 'rV3v', 'lV3d', 'rV3d'],
            'hV4': ['lhV4', 'rhV4'],  # Note: hV4 is used for V4
            'BMD': ['BMDgeneral'],
        }

        if subjects is None:
            self.subjects = self._available_subjects
        else:
            self.subjects = self._validate_subjects(subjects)

        self.roi_classes = rois
        self.rois = []  # Keep for consistency
        for r in rois:
            self.rois.extend(self.regions_map[r])
        if self.rois is None:
            self.rois = self._get_available_rois()

        self.overwrite = overwrite
        self.data: Dict[int, Dict[str, np.ndarray]] = {}  # Store loaded data

    def _validate_subjects(self, subjects: List[int]) -> List[int]:
        for sub in subjects:
            if not isinstance(sub, int) or sub not in self._available_subjects:
                raise ValueError(
                    f"Invalid subject ID: {sub}.  Must be in 1-10."
                )
        return subjects

    def _validate_rois(self, rois: List[str]) -> List[str]:
        if self._available_rois is None:
            self._get_available_rois()
        if rois is None:
            return self._available_rois
        for roi in rois:
            if roi not in self._available_rois:
                raise ValueError(f"Invalid ROI: {roi}")
        return rois

    def _get_available_rois(self) -> List[str]:
        if self._available_rois is not None:
            return self._available_rois

        if not self.is_downloaded():
            self.fetch_and_extract()

        first_subject_dir = os.path.join(
            self.root_dir, f"sub-{self.subjects[0]:02d}", "ROIs")
        available_rois = []
        roi_pattern = re.compile(
            r"ROI-(?P<roi>[^_]+)_indices\.pkl"
        )
        for filename in os.listdir(first_subject_dir):
            match = roi_pattern.match(filename)
            if match:
                available_rois.append(match.group("roi"))
        self._available_rois = available_rois
        return self._available_rois

    def is_downloaded(self) -> bool:
        for sub_id in self.subjects:
            subject_dir = os.path.join(self.root_dir, f"sub-{sub_id:02d}")
            if not os.path.exists(subject_dir):
                return False
        return True

    def _check_prepared_betas_downloaded(self, subject: int) -> bool:
        """
        Checks if all required prepared betas files exist for the given subject.
        """
        sub_folder = f"sub-{subject:02d}"
        required_files = [
            f"sub-{subject:02d}_organized_betas_task-train_normalized.pkl",
            f"sub-{subject:02d}_organized_betas_task-test_normalized.pkl",
            f"sub-{subject:02d}_noiseceiling_task-train_n-3.pkl",
            f"sub-{subject:02d}_noiseceiling_task-test_n-10.pkl"
        ]
        for file_name in required_files:
            file_path = os.path.join(
                self.root_dir, sub_folder, "prepared_betas", file_name)
            if not os.path.exists(file_path):
                return False
        return True

    def _check_rois_downloaded(self, subject: int) -> bool:
        """
        Checks if all required ROI files exist for the given subject.
        """
        sub_folder = f"sub-{subject:02d}"
        for region, rois in self.regions_map.items():
            for roi in rois:
                file_path = os.path.join(
                    self.root_dir, sub_folder, "ROIs", f"ROI-{roi}_indices.pkl")
                if not os.path.exists(file_path):
                    return False
        return True

    def fetch_and_extract(self, force_download: bool = False, **kwargs) -> list:
        """
        Downloads and extracts data for all subjects using BaseDataset's methods.

        For each subject, if the required files in the ROIs and prepared_betas directories
        already exist (unless force_download is True), then the download is skipped.
        """
        target_dir = self.root_dir
        data_paths = []

        for subject in self.subjects:
            sub_folder = f"sub-{subject:02d}"
            sub_dir = os.path.join(target_dir, sub_folder)

            # Check whether the required files already exist.
            prepared_betas_exist = self._check_prepared_betas_downloaded(
                subject)
            rois_exist = self._check_rois_downloaded(subject)

            if prepared_betas_exist and rois_exist and not force_download:
                print(f"Data already downloaded for {sub_folder}")
                data_paths.append(sub_dir)
                continue

            print(f"Downloading fMRI, NCSNR, and ROI data for {sub_folder}...")
            source_base = os.path.join(self.url, sub_folder)

            try:
                # Download the ROIs folder if needed.
                if not rois_exist or force_download:
                    rois_url = os.path.join(source_base, "ROIs")
                    self.fetch(
                        source=rois_url,
                        target_dir=sub_dir,
                        method='s3',
                        anonymous=True,
                        force_download=force_download,
                        progress_bar=True
                    )

                # Download the prepared_betas folder if needed.
                if not prepared_betas_exist or force_download:
                    betas_url = os.path.join(source_base, "prepared_betas")
                    self.fetch(
                        source=betas_url,
                        target_dir=sub_dir,
                        method='s3',
                        anonymous=True,
                        force_download=force_download,
                        progress_bar=True
                    )

                # Verify that the folders now exist.
                if not os.path.exists(os.path.join(sub_dir, "ROIs")):
                    raise RuntimeError(
                        f"ROIs directory missing for {sub_folder}")
                if not os.path.exists(os.path.join(sub_dir, "prepared_betas")):
                    raise RuntimeError(
                        f"prepared_betas directory missing for {sub_folder}")

                # Confirm that all required files are present.
                if not self._check_prepared_betas_downloaded(subject):
                    raise RuntimeError(
                        f"Not all prepared betas files are present for {sub_folder}")
                if not self._check_rois_downloaded(subject):
                    raise RuntimeError(
                        f"Not all ROI files are present for {sub_folder}")

                data_paths.append(sub_dir)

            except Exception as e:
                raise RuntimeError(
                    f"Download failed for {sub_folder}: {e}") from e

        return data_paths

    def _load_data(self, subject: int, roi: str) -> np.ndarray:
        filepath_train = os.path.join(self.root_dir,
                                      f"sub-{subject:02d}", "prepared_betas", f"sub-{subject:02d}_organized_betas_task-train_normalized.pkl"
                                      )

        filepath_test = os.path.join(self.root_dir,
                                     f"sub-{subject:02d}", "prepared_betas", f"sub-{subject:02d}_organized_betas_task-test_normalized.pkl"
                                     )

        if not os.path.exists(filepath_train) or not os.path.exists(filepath_test):
            self.fetch_and_extract()
            if not os.path.exists(filepath_train):
                raise FileNotFoundError(f"File not found: {filepath_train}")

        # Check for the required ncsnr files
        train_ncsnr = os.path.join(self.root_dir,
                                   f"sub-{subject:02d}", "prepared_betas", f"sub-{subject:02d}_noiseceiling_task-train_n-3.pkl")
        test_ncsnr = os.path.join(self.root_dir,
                                  f"sub-{subject:02d}", "prepared_betas", f"sub-{subject:02d}_noiseceiling_task-test_n-10.pkl")

        if not os.path.exists(train_ncsnr) or not os.path.exists(test_ncsnr):
            raise RuntimeError(
                f"Noise ceiling files missing for sub-{subject:02d}")

        # ROI
        roi_path = os.path.join(self.root_dir,
                                f"sub-{subject:02d}", "ROIs", f"ROI-{roi}_indices.pkl")

        if not os.path.exists(train_ncsnr) or not os.path.exists(test_ncsnr):
            raise RuntimeError(f"ROI files missing for sub-{subject:02d}")

        # Get ROI specific data
        with open(roi_path, 'rb') as f:
            roi_indices = pickle.load(f)[1].flatten()

        with open(filepath_train, 'rb') as f:
            fmri_data_train, _, _ = pickle.load(f)
        betas_train = fmri_data_train[:, :, roi_indices]

        with open(filepath_test, 'rb') as f:
            fmri_data_test, _, _ = pickle.load(f)
        betas_test = fmri_data_test[:, :, roi_indices]

        with open(train_ncsnr, 'rb') as f:
            train_ncsnr = pickle.load(f)  # TODO: make sure you divide by 100
            ncsnr, ceiling = train_ncsnr[0], train_ncsnr[1]
            ceiling = ceiling[roi_indices] / 100

        with open(test_ncsnr, 'rb') as f:
            test_ncsnr = pickle.load(f)
            ncsnr_test, ceiling_test = test_ncsnr[0], test_ncsnr[1]
            ceiling_test = ceiling_test[roi_indices] / 100

        # Find ROI indices where both training and test ncsnr values are above 0.2
        valid_roi = np.where((ceiling > NCSNR_THRESHOLD) &
                             (ceiling_test > NCSNR_THRESHOLD))[0]
        # valid_roi = np.where(ceiling_test > NCSNR_THRESHOLD)[0]

        # Optionally, update your betas and ncsnr arrays to include only these indices:
        betas_train = betas_train[:, :, valid_roi]
        betas_test = betas_test[:, :, valid_roi]
        ceiling = ceiling[valid_roi]
        ceiling_test = ceiling_test[valid_roi]

        return betas_train.mean(axis=1), betas_test.mean(axis=1), ceiling, ceiling_test

    def _prepare_subject_data(self, subject: int):
        """Pre-loads all ROIs for a given subject."""
        if subject not in self.data:
            self.data[subject] = {}
            for roi in self.rois:
                self.data[subject][roi] = self._load_data(subject, roi)

    def _combine_subjects_within_roi(self, roi: str) -> np.ndarray:
        """Combines data across subjects *within* a single ROI."""
        roi_train, roi_test, ceiling = [], [], []
        for subject in self.subjects:
            # Ensure all subject data loaded
            self._prepare_subject_data(subject)
            if roi in self.data[subject]:  # Handle missing ROIs
                betas_train, betas_test, _, ceiling_t = self.data[subject][roi]
                roi_train.append(betas_train)
                roi_test.append(betas_test)
                ceiling.append(ceiling_t)
        return np.concatenate(roi_train, axis=1), np.concatenate(roi_test, axis=1),  np.concatenate(ceiling)

    def get_assembly(self, train: bool = True) -> np.ndarray:
        region_path = "_".join(sorted(self.roi_classes))
        if self._check_assembly_exists(region_path, train):
            return self._load_assembly(region_path, train)

        print("Generate assemblies from raw files.")
        combined_regions_data_train,  combined_regions_data_test, ceiling = [], [], []
        for region in self.roi_classes:
            if region not in self.regions_map:
                print(
                    f"Warning: Region '{region}' not found in regions_map. Skipping.")
                continue

            rois_in_region = [
                roi for roi in self._get_available_rois()
                if any(roi.endswith(mapped_roi) or roi.startswith(mapped_roi) for mapped_roi in self.regions_map[region])
            ]

            if not rois_in_region:
                print(
                    f"Warning: No ROIs found for region '{region}'. Skipping.")
                continue

            region_train, region_test, region_ceiling = [], [], []
            for roi in rois_in_region:
                combined_train_data, combined_test_data, combined_ceiling = self._combine_subjects_within_roi(
                    roi)
                region_train.append(combined_train_data)
                region_test.append(combined_test_data)
                region_ceiling.append(combined_ceiling)

            train_assembly_roi = np.concatenate(region_train, axis=1)
            test_assembly_roi = np.concatenate(region_test, axis=1)

            combined_regions_data_train.append(train_assembly_roi)
            combined_regions_data_test.append(test_assembly_roi)
            ceiling.append(np.concatenate(region_ceiling))

            if train:
                print(f"Size of area {region}: {train_assembly_roi.shape}")
            else:
                print(f"Size of area {region}: {test_assembly_roi.shape}")

        if train:
            train_assembly = np.concatenate(
                combined_regions_data_train, axis=1)
            ceiling = np.concatenate(ceiling)
            self._save_assemblies(train_assembly, ceiling, region_path, train)
            return train_assembly, ceiling

        test_assembly = np.concatenate(combined_regions_data_test, axis=1)
        ceiling = np.concatenate(ceiling)
        self._save_assemblies(test_assembly, ceiling, region_path, train)
        return test_assembly, ceiling

    def _save_assemblies(self, data, ceiling, region, train):
        # Define file paths for train and test assemblies and ceilings
        assembly_path = os.path.join(
            self.root_dir, region, 'train_assembly.pkl' if train else 'test_assembly.pkl')
        ceiling_path = os.path.join(
            self.root_dir, region, 'ceiling_train.pkl' if train else 'ceiling_test.pkl')

        if not os.path.exists(os.path.dirname(assembly_path)):
            os.makedirs(os.path.dirname(assembly_path), exist_ok=True)

        # Save the train data
        with open(assembly_path, 'wb') as f:
            pickle.dump(data, f)
        with open(ceiling_path, 'wb') as f:
            pickle.dump(ceiling, f)

    def _check_assembly_exists(self, region, train):
        assembly_file = os.path.join(
            self.root_dir, region, 'train_assembly.pkl' if train else 'test_assembly.pkl')
        ceiling_file = os.path.join(
            self.root_dir, region, 'ceiling_train.pkl' if train else 'ceiling_test.pkl')
        return os.path.exists(assembly_file) and os.path.exists(ceiling_file)

    def _load_assembly(self, region, train):
        assembly_file = os.path.join(
            self.root_dir, region, 'train_assembly.pkl' if train else 'test_assembly.pkl')
        ceiling_file = os.path.join(
            self.root_dir, region, 'ceiling_train.pkl' if train else 'ceiling_test.pkl')
        with open(assembly_file, 'rb') as f:
            assembly = pickle.load(f)
        with open(ceiling_file, 'rb') as f:
            ceiling = pickle.load(f)
        print("Loaded assemblies from pickle files.")
        return assembly, ceiling

    def __len__(self) -> int:
        return len(self.subjects) * len(self._get_available_rois())

    def __getitem__(self, idx: int) -> Tuple[int, str, np.ndarray]:
        # this is a dummy function for now
        rois = self.rois
        num_rois = len(rois)
        subject_index = idx // num_rois
        roi_index = idx % num_rois
        subject = self.subjects[subject_index]
        roi = rois[roi_index]
        self._prepare_subject_data(subject)  # Load all ROIs for the subject
        return subject, roi, self.data[subject][roi]


class BMDAssemblyV1(BMDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), BMDAssembly.__name__
        )
        super().__init__(root_dir=root, rois=['V1'], **kwargs)


class BMDAssemblyV2(BMDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), BMDAssembly.__name__
        )
        super().__init__(root_dir=root, rois=['V2'], **kwargs)


class BMDAssemblyV3(BMDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), BMDAssembly.__name__
        )
        super().__init__(root_dir=root, rois=['V3'], **kwargs)


class BMDAssemblyV4(BMDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), BMDAssembly.__name__
        )
        super().__init__(root_dir=root, rois=['hV4'], **kwargs)


class BMDAssemblyIPS0(BMDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), BMDAssembly.__name__
        )
        super().__init__(root_dir=root, rois=['IPS0'], **kwargs)


class BMDAssemblyIPS123(BMDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), BMDAssembly.__name__
        )
        super().__init__(root_dir=root, rois=['IPS1-2-3'], **kwargs)


class BMDAssemblyLOC(BMDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), BMDAssembly.__name__
        )
        super().__init__(root_dir=root, rois=['LOC'], **kwargs)


class BMDAssemblyPFop(BMDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), BMDAssembly.__name__
        )
        super().__init__(root_dir=root, rois=['PFop'], **kwargs)


class BMDAssembly7AL(BMDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), BMDAssembly.__name__
        )
        super().__init__(root_dir=root, rois=['7AL'], **kwargs)


class BMDAssemblyPFt(BMDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), BMDAssembly.__name__
        )
        super().__init__(root_dir=root, rois=['PFt'], **kwargs)


class BMDAssemblyOFA(BMDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), BMDAssembly.__name__
        )
        super().__init__(root_dir=root, rois=['OFA'], **kwargs)


class BMDAssemblyBA2(BMDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), BMDAssembly.__name__
        )
        super().__init__(root_dir=root, rois=['BA2'], **kwargs)


class BMDAssemblyEBA(BMDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), BMDAssembly.__name__
        )
        super().__init__(root_dir=root, rois=['EBA'], **kwargs)


class BMDAssemblyFFA(BMDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), BMDAssembly.__name__
        )
        super().__init__(root_dir=root, rois=['FFA'], **kwargs)


class BMDAssemblyMT(BMDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), BMDAssembly.__name__
        )
        super().__init__(root_dir=root, rois=['MT'], **kwargs)


class BMDAssemblyPPA(BMDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), BMDAssembly.__name__
        )
        super().__init__(root_dir=root, rois=['PPA'], **kwargs)


class BMDAssemblyRSC(BMDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), BMDAssembly.__name__
        )
        super().__init__(root_dir=root, rois=['RSC'], **kwargs)


class BMDAssemblySTS(BMDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), BMDAssembly.__name__
        )
        super().__init__(root_dir=root, rois=['STS'], **kwargs)


class BMDAssemblyTOS(BMDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), BMDAssembly.__name__
        )
        super().__init__(root_dir=root, rois=['TOS'], **kwargs)


class BMDAssemblyV3ab(BMDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), BMDAssembly.__name__
        )
        super().__init__(root_dir=root, rois=['V3ab'], **kwargs)


class BMDAssemblyV1d(BMDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), BMDAssembly.__name__
        )
        super().__init__(root_dir=root, rois=['V1d'], **kwargs)


class BMDAssemblyV1v(BMDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), BMDAssembly.__name__
        )
        super().__init__(root_dir=root, rois=['V1v'], **kwargs)


class BMDAssemblyV2d(BMDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), BMDAssembly.__name__
        )
        super().__init__(root_dir=root, rois=['V2d'], **kwargs)


class BMDAssemblyV2v(BMDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), BMDAssembly.__name__
        )
        super().__init__(root_dir=root, rois=['V2v'], **kwargs)


class BMDAssemblyV3d(BMDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), BMDAssembly.__name__
        )
        super().__init__(root_dir=root, rois=['V3d'], **kwargs)


class BMDAssemblyV3v(BMDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), BMDAssembly.__name__
        )
        super().__init__(root_dir=root, rois=['V3v'], **kwargs)


class BMDAssemblyBMD(BMDAssembly):
    def __init__(self, **kwargs):
        root = os.path.join(
            get_data_home(), BMDAssembly.__name__
        )
        super().__init__(root_dir=root, rois=['V2v'], **kwargs)
