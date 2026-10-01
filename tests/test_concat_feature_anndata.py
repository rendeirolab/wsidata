import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from wsidata import io, open_wsi
from wsidata.io._wsi import _concat_feature_anndata


class TestConcatFeatureAnnData:
    """Tests for the concat_feature_anndata function.

    The concat_feature_anndata function aggregates features from multiple slides
    into an AnnData or AnnCollection object. These tests verify that the function
    works correctly with different parameters and handles error cases appropriately.

    The tests use mocking to isolate the testing of the concat_feature_anndata function
    from the complexities of the zarr file structure, making them more reliable and focused.
    """

    @patch("wsidata.io._wsi._concat_feature_anndata")
    def test_concat_feature_anndata_basic_store_col(
        self, mock_concat_feature_anndata, tmpdir
    ):
        """Test basic functionality of concat_feature_anndata with store_col parameter."""
        # Create mock AnnData objects
        n_obs = 100
        n_vars = 50
        mock_adata1 = AnnData(
            X=np.random.rand(n_obs, n_vars),
            obs=pd.DataFrame({"cell_id": [f"cell_{i}" for i in range(n_obs)]}),
        )
        mock_adata2 = AnnData(
            X=np.random.rand(n_obs, n_vars),
            obs=pd.DataFrame(
                {"cell_id": [f"cell_{i}" for i in range(n_obs, 2 * n_obs)]}
            ),
        )

        # Mock the _concat_feature_anndata function to return AnnData objects
        mock_concat_feature_anndata.side_effect = [mock_adata1, mock_adata2]

        # Create a slides table with dummy store paths
        store_path1 = Path(tmpdir) / "slide1.zarr"
        store_path2 = Path(tmpdir) / "slide2.zarr"
        slides_table = pd.DataFrame(
            {
                "slide_id": ["slide1", "slide2"],
                "store_path": [str(store_path1), str(store_path2)],
            }
        )

        # Test concat_feature_anndata with store_col
        result = io.concat_feature_anndata(
            slides_table=slides_table,
            feature_key="test_feature",
            tile_key="tiles",
            store_col="store_path",
        )

        # Check that _concat_feature_anndata was called twice with correct arguments
        assert mock_concat_feature_anndata.call_count == 2
        calls = mock_concat_feature_anndata.call_args_list
        assert str(store_path1) in [call[0][0] for call in calls]
        assert str(store_path2) in [call[0][0] for call in calls]

        # Check that all calls had the expected feature_key, tile_key, and error args
        for call in calls:
            assert call[0][1] == "test_feature"
            assert call[0][2] == "tiles"
            assert call[0][3] == "raise"

        # Check that the result is an AnnData object
        assert isinstance(result, AnnData)
        # Check that the result has the expected shape (concatenated)
        assert result.X.shape[0] == 2 * n_obs  # Two slides concatenated
        assert result.X.shape[1] == n_vars
        # Check that slide_name label was added
        assert "slide_name" in result.obs.columns

    @patch("wsidata.io._wsi._concat_feature_anndata")
    def test_concat_feature_anndata_basic_wsi_col(
        self, mock_concat_feature_anndata, tmpdir
    ):
        """Test basic functionality of concat_feature_anndata with wsi_col parameter."""
        # Create mock AnnData objects
        n_obs = 100
        n_vars = 50
        mock_adata = AnnData(
            X=np.random.rand(n_obs, n_vars),
            obs=pd.DataFrame({"cell_id": [f"cell_{i}" for i in range(n_obs)]}),
        )

        # Mock the _concat_feature_anndata function to return AnnData object
        mock_concat_feature_anndata.return_value = mock_adata

        # Create a slides table with WSI paths (will be converted to .zarr)
        wsi_path = Path(tmpdir) / "slide1.svs"
        slides_table = pd.DataFrame(
            {"slide_id": ["slide1"], "wsi_path": [str(wsi_path)]}
        )

        # Test concat_feature_anndata with wsi_col
        result = io.concat_feature_anndata(
            slides_table=slides_table,
            feature_key="test_feature",
            tile_key="tiles",
            wsi_col="wsi_path",
        )

        # Check that _concat_feature_anndata was called with the .zarr path
        mock_concat_feature_anndata.assert_called_once()
        args = mock_concat_feature_anndata.call_args[0]
        # Convert both to Path objects for comparison to handle path resolution
        expected_path = Path(str(wsi_path.with_suffix(".zarr")))
        actual_path = Path(args[0])
        assert actual_path == expected_path  # Should convert to .zarr
        assert args[1] == "test_feature"
        assert args[2] == "tiles"
        assert args[3] == "raise"

        # Check that the result is an AnnData object
        assert isinstance(result, AnnData)

    @patch("wsidata.io._wsi._concat_feature_anndata")
    def test_concat_feature_anndata_as_anncollection(
        self, mock_concat_feature_anndata, tmpdir
    ):
        """Test concat_feature_anndata with as_anncollection=True."""
        # Create mock AnnData objects
        n_obs = 100
        n_vars = 50
        mock_adata1 = AnnData(
            X=np.random.rand(n_obs, n_vars),
            obs=pd.DataFrame({"cell_id": [f"cell_{i}" for i in range(n_obs)]}),
        )
        mock_adata2 = AnnData(
            X=np.random.rand(n_obs, n_vars),
            obs=pd.DataFrame(
                {"cell_id": [f"cell_{i}" for i in range(n_obs, 2 * n_obs)]}
            ),
        )

        # Mock the _concat_feature_anndata function to return AnnData objects
        mock_concat_feature_anndata.side_effect = [mock_adata1, mock_adata2]

        # Create a slides table with dummy store paths
        store_path1 = Path(tmpdir) / "slide1.zarr"
        store_path2 = Path(tmpdir) / "slide2.zarr"
        slides_table = pd.DataFrame(
            {
                "slide_id": ["slide1", "slide2"],
                "store_path": [str(store_path1), str(store_path2)],
            }
        )

        # Test concat_feature_anndata with as_anncollection=True
        result = io.concat_feature_anndata(
            slides_table=slides_table,
            feature_key="test_feature",
            tile_key="tiles",
            store_col="store_path",
            as_anncollection=True,
        )

        # Check that the result is an AnnCollection object
        from anndata.experimental import AnnCollection

        assert isinstance(result, AnnCollection)
        # Check that it contains the expected number of slides (collections)
        # AnnCollection stores individual AnnData objects, so we check the dict length
        assert len(result.adatas) == 2

    def test_concat_feature_anndata_no_col_error(self, tmpdir):
        """Test that ValueError is raised when neither wsi_col nor store_col is provided."""
        slides_table = pd.DataFrame(
            {"slide_id": ["slide1"], "some_other_col": ["value1"]}
        )

        with pytest.raises(
            ValueError, match="Either wsi_col or store_col must be provided"
        ):
            io.concat_feature_anndata(
                slides_table=slides_table, feature_key="test_feature", tile_key="tiles"
            )

    @patch("wsidata.io._wsi._concat_feature_anndata")
    def test_concat_feature_anndata_error_raise_mode(
        self, mock_concat_feature_anndata, tmpdir
    ):
        """Test concat_feature_anndata with error='raise' when an exception occurs."""
        # Mock the _concat_feature_anndata function to raise an exception
        mock_concat_feature_anndata.side_effect = ValueError("File not found")

        # Create a slides table with dummy store path
        store_path = Path(tmpdir) / "nonexistent.zarr"
        slides_table = pd.DataFrame(
            {"slide_id": ["slide1"], "store_path": [str(store_path)]}
        )

        # Test that the exception is propagated when error='raise'
        with pytest.raises(ValueError, match="File not found"):
            io.concat_feature_anndata(
                slides_table=slides_table,
                feature_key="test_feature",
                tile_key="tiles",
                store_col="store_path",
                error="raise",
            )

    @patch("wsidata.io._wsi._concat_feature_anndata")
    def test_concat_feature_anndata_error_skip_mode(
        self, mock_concat_feature_anndata, tmpdir
    ):
        """Test concat_feature_anndata with error='skip' when some files fail."""
        # Create mock AnnData object for successful case
        n_obs = 100
        n_vars = 50
        mock_adata = AnnData(
            X=np.random.rand(n_obs, n_vars),
            obs=pd.DataFrame({"cell_id": [f"cell_{i}" for i in range(n_obs)]}),
        )

        # Mock: first call succeeds, second call returns None (simulating skip)
        mock_concat_feature_anndata.side_effect = [mock_adata, None]

        # Create slides table with two store paths
        store_path1 = Path(tmpdir) / "slide1.zarr"
        store_path2 = Path(tmpdir) / "slide2.zarr"  # This one will "fail"
        slides_table = pd.DataFrame(
            {
                "slide_id": ["slide1", "slide2"],
                "store_path": [str(store_path1), str(store_path2)],
            }
        )

        # None results are filtered out; only the successful slide is concatenated
        result = io.concat_feature_anndata(
            slides_table=slides_table,
            feature_key="test_feature",
            tile_key="tiles",
            store_col="store_path",
            error="skip",
        )

        # Check that _concat_feature_anndata was called twice
        assert mock_concat_feature_anndata.call_count == 2
        # Only the successful slide should be in the result
        assert isinstance(result, AnnData)
        assert result.X.shape[0] == n_obs
        assert result.X.shape[1] == n_vars

    @patch("wsidata.io._wsi._concat_feature_anndata")
    def test_concat_feature_anndata_multiple_slides(
        self, mock_concat_feature_anndata, tmpdir
    ):
        """Test concat_feature_anndata with multiple slides."""
        # Create multiple mock AnnData objects
        n_slides = 5
        n_obs = 50
        n_vars = 30

        mock_adatas = []
        for i in range(n_slides):
            mock_adata = AnnData(
                X=np.random.rand(n_obs, n_vars),
                obs=pd.DataFrame(
                    {"cell_id": [f"slide{i}_cell_{j}" for j in range(n_obs)]}
                ),
            )
            mock_adatas.append(mock_adata)

        mock_concat_feature_anndata.side_effect = mock_adatas

        # Create slides table with multiple store paths
        slides_table = pd.DataFrame(
            {
                "slide_id": [f"slide{i}" for i in range(n_slides)],
                "store_path": [
                    str(Path(tmpdir) / f"slide{i}.zarr") for i in range(n_slides)
                ],
            }
        )

        # Test concat_feature_anndata with multiple slides
        result = io.concat_feature_anndata(
            slides_table=slides_table,
            feature_key="test_feature",
            tile_key="tiles",
            store_col="store_path",
        )

        # Check that _concat_feature_anndata was called for each slide
        assert mock_concat_feature_anndata.call_count == n_slides

        # Check that the result has the expected shape
        assert isinstance(result, AnnData)
        assert result.X.shape[0] == n_slides * n_obs  # All slides concatenated
        assert result.X.shape[1] == n_vars
        # Check that slide_name label was added
        assert "slide_name" in result.obs.columns

    @patch("wsidata.io._wsi._concat_feature_anndata")
    def test_concat_feature_anndata_empty_table(self, mock_concat_feature_anndata):
        """Test concat_feature_anndata with empty slides_table."""
        # Create empty slides table
        slides_table = pd.DataFrame({"slide_id": [], "store_path": []})

        # Empty table → no slides → returns empty AnnData with warning
        with pytest.warns(UserWarning, match="No slides produced valid features"):
            result = io.concat_feature_anndata(
                slides_table=slides_table,
                feature_key="test_feature",
                tile_key="tiles",
                store_col="store_path",
            )

        assert isinstance(result, AnnData)
        assert result.n_obs == 0

        # _concat_feature_anndata should not be called at all
        mock_concat_feature_anndata.assert_not_called()

    @patch("wsidata.io._wsi._concat_feature_anndata")
    @patch("wsidata.io._wsi.track")
    def test_concat_feature_anndata_pbar(
        self, mock_track, mock_concat_feature_anndata, tmpdir
    ):
        """Test concat_feature_anndata with progress bar enabled."""
        # Create mock AnnData object
        n_obs = 100
        n_vars = 50
        mock_adata = AnnData(
            X=np.random.rand(n_obs, n_vars),
            obs=pd.DataFrame({"cell_id": [f"cell_{i}" for i in range(n_obs)]}),
        )

        mock_concat_feature_anndata.return_value = mock_adata
        # Mock track to return the iterator as-is
        mock_track.side_effect = lambda x, **kwargs: x

        # Create a slides table
        store_path = Path(tmpdir) / "slide1.zarr"
        slides_table = pd.DataFrame(
            {"slide_id": ["slide1"], "store_path": [str(store_path)]}
        )

        # Test with pbar=True
        result = io.concat_feature_anndata(
            slides_table=slides_table,
            feature_key="test_feature",
            tile_key="tiles",
            store_col="store_path",
            pbar=True,
        )

        # Check that track was called with disable=False
        mock_track.assert_called_once()
        call_kwargs = mock_track.call_args[1]
        assert not call_kwargs["disable"]
        assert "Concatenation of 1 slides" in call_kwargs["description"]

        # Check result
        assert isinstance(result, AnnData)

    @pytest.mark.parametrize("as_anncollection", [False, True])
    @patch("wsidata.io._wsi._concat_feature_anndata")
    def test_concat_feature_anndata_slide_name(
        self, mock_concat_feature_anndata, tmp_path, as_anncollection
    ):
        """Regression: the slides were labelled by their row, so
        obs["slide_name"] held 0, 1, ... and the obs names ended in -0, -1, ...
        instead of the name of the slide, the name of its store without .zarr.
        """
        mock_concat_feature_anndata.side_effect = lambda *args: AnnData(
            np.random.rand(2, 5)
        )
        slides_table = pd.DataFrame(
            {
                "store_path": [
                    str(tmp_path / "slide1.zarr"),
                    str(tmp_path / "slide2.scene-1.zarr"),
                ]
            }
        )

        result = io.concat_feature_anndata(
            slides_table=slides_table,
            feature_key="test_feature",
            tile_key="tiles",
            store_col="store_path",
            as_anncollection=as_anncollection,
        )

        assert result.obs["slide_name"].tolist() == [
            "slide1",
            "slide1",
            "slide2.scene-1",
            "slide2.scene-1",
        ]
        assert list(result.obs_names) == [
            "0-slide1",
            "1-slide1",
            "0-slide2.scene-1",
            "1-slide2.scene-1",
        ]

    @patch("wsidata.io._wsi._concat_feature_anndata")
    def test_concat_feature_anndata_duplicate_slide_names(
        self, mock_concat_feature_anndata, tmp_path
    ):
        """Regression: as above; named by their stores, two slides whose stores
        share a name, in different folders, would get one name, and one of the
        slides would be lost. It raises, naming them, before any store is read.
        """
        mock_concat_feature_anndata.side_effect = lambda *args: AnnData(
            np.random.rand(2, 5)
        )
        slides_table = pd.DataFrame(
            {
                "store_path": [
                    str(tmp_path / "a" / "slide1.zarr"),
                    str(tmp_path / "slide2.zarr"),
                    str(tmp_path / "b" / "slide1.zarr"),
                ]
            }
        )
        for store in slides_table["store_path"]:
            Path(store).mkdir(parents=True)

        with pytest.raises(ValueError, match="slide1") as excinfo:
            io.concat_feature_anndata(
                slides_table=slides_table,
                feature_key="test_feature",
                tile_key="tiles",
                store_col="store_path",
            )
        assert "slide2" not in str(excinfo.value)
        mock_concat_feature_anndata.assert_not_called()

    @patch("wsidata.io._wsi._concat_feature_anndata")
    def test_concat_feature_anndata_skips_missing_stores_sharing_a_name(
        self, mock_concat_feature_anndata, tmp_path
    ):
        """Regression: as above; the stores that do not exist cannot collide, as
        none is read. With error="skip", empty store_col cells, which all name
        the store "nan", are skipped as before, not raised as duplicates.
        """
        mock_concat_feature_anndata.side_effect = lambda f, *args: (
            AnnData(np.random.rand(2, 5)) if Path(f).exists() else None
        )
        (tmp_path / "slide1.zarr").mkdir()
        slides_table = pd.DataFrame(
            {"store_path": [str(tmp_path / "slide1.zarr"), np.nan, np.nan]}
        )

        result = io.concat_feature_anndata(
            slides_table=slides_table,
            feature_key="test_feature",
            tile_key="tiles",
            store_col="store_path",
            error="skip",
        )

        assert result.obs["slide_name"].unique().tolist() == ["slide1"]


class TestConcatFeatureAnnDataHelper:
    """Tests for the _concat_feature_anndata helper function."""

    @patch("pathlib.Path.exists")
    def test_concat_feature_anndata_helper_file_not_exists_raise(self, mock_exists):
        """Test _concat_feature_anndata with non-existent file and error='raise'."""
        mock_exists.return_value = False

        with pytest.raises(ValueError, match="File .* not existed"):
            _concat_feature_anndata(
                "nonexistent.zarr", "test_feature", "tiles", error="raise"
            )

    @patch("pathlib.Path.exists")
    def test_concat_feature_anndata_helper_file_not_exists_skip(self, mock_exists):
        """Test _concat_feature_anndata with non-existent file and error='skip'."""
        mock_exists.return_value = False

        result = _concat_feature_anndata(
            "nonexistent.zarr", "test_feature", "tiles", error="skip"
        )

        assert result is None

    @patch("pathlib.Path.exists")
    @patch("zarr.open")
    @patch("anndata.read_zarr")
    def test_concat_feature_anndata_helper_success(
        self, mock_read_zarr, mock_zarr_open, mock_exists
    ):
        """Test _concat_feature_anndata with successful execution."""
        mock_exists.return_value = True

        # Mock zarr structure
        mock_tables = MagicMock()
        mock_tables.keys.return_value = ["test_feature_tiles"]
        mock_zarr_open.return_value = mock_tables

        # Mock AnnData object
        mock_adata = AnnData(
            X=np.random.rand(100, 50),
            obs=pd.DataFrame({"cell_id": [f"cell_{i}" for i in range(100)]}),
        )
        mock_read_zarr.return_value = mock_adata

        result = _concat_feature_anndata(
            "test.zarr", "test_feature", "tiles", error="raise"
        )

        # Check that zarr.open was called correctly
        mock_zarr_open.assert_called_once_with("test.zarr/tables")
        # Check that read_zarr was called correctly
        mock_read_zarr.assert_called_once_with("test.zarr/tables/test_feature_tiles")
        # Check that result is the expected AnnData object
        assert result is mock_adata

    @patch("pathlib.Path.exists")
    @patch("zarr.open")
    def test_concat_feature_anndata_helper_exception_raise(
        self, mock_zarr_open, mock_exists
    ):
        """Test _concat_feature_anndata with exception and error='raise'."""
        mock_exists.return_value = True
        mock_zarr_open.side_effect = Exception("Zarr error")

        with pytest.raises(Exception, match="Zarr error"):
            _concat_feature_anndata("test.zarr", "test_feature", "tiles", error="raise")

    @patch("pathlib.Path.exists")
    @patch("zarr.open")
    def test_concat_feature_anndata_helper_exception_skip(
        self, mock_zarr_open, mock_exists
    ):
        """Test _concat_feature_anndata with exception and error='skip'."""
        mock_exists.return_value = True
        mock_zarr_open.side_effect = Exception("Zarr error")

        result = _concat_feature_anndata(
            "test.zarr", "test_feature", "tiles", error="skip"
        )

        assert result is None


def test_concat_feature_anndata_open_wsi_scene_store(test_multiscene_czi, tmp_path):
    """Regression: with wsi_col, concat_feature_anndata looked for the store of
    a scene of a multi-scene slide at <stem>.zarr, not where open_wsi writes
    it, and named the slide 0 in obs["slide_name"].
    """
    # Two suffixes: open_wsi names the store by the stem, "multi_scene.ome"
    slide = tmp_path / "multi_scene.ome.czi"
    shutil.copy(test_multiscene_czi, slide)
    wsi = open_wsi(slide, reader="pylibczi", scene=1)
    wsi.tables["test_feature_tiles"] = AnnData(np.ones((3, 4)))
    wsi.write()
    wsi.close()

    result = io.concat_feature_anndata(
        slides_table=pd.DataFrame({"wsi_path": [str(slide)]}),
        feature_key="test_feature",
        tile_key="tiles",
        wsi_col="wsi_path",
    )

    assert result.obs["slide_name"].tolist() == ["multi_scene.ome.scene-1"] * 3
