import gi
import sys
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject

def display_video():
    Gst.init(None)
    
    # Create the GStreamer pipeline
    pipeline_description = 'v4l2src ! videoconvert ! autovideosink'
    pipeline = Gst.parse_launch(pipeline_description)

    # Start playing
    pipeline.set_state(Gst.State.PLAYING)

    # Wait until error or EOS (End of Stream)
    bus = pipeline.get_bus()
    msg = bus.timed_pop_filtered(
        Gst.CLOCK_TIME_NONE,
        Gst.MessageType.ERROR | Gst.MessageType.EOS
    )

    # Free resources
    pipeline.set_state(Gst.State.NULL)

display_video()
