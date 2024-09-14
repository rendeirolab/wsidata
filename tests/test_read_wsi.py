from wsidata import open_wsi


def test_read_local(test_slide):
    open_wsi(test_slide)


def test_read_url(tmpdir):
    slide_url = "https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1-Small-Region.svs"
    # Download the slide to a temporary directory
    open_wsi(slide_url, cache_dir=tmpdir)
