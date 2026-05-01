from typing import Union

import pretty_midi
import mido
import six
import warnings
import numpy as np
import collections
CHORD_MAP = [
    'maj', 'maj6', 'maj7', 'maj7(#11)', 'maj(9)', 'maj7(9)', 'maj6(9)', 'aug', 'min', 'min6', 'min7', 'hdim7', 'min(9)',
    'min7(9)', 'min7(11)', 'minmaj7', 'minmaj7(9)', 'dim', 'dim7', '7', 'sus4(b7)', '(3,b5,b7)', '7(9)', '7(#11)', '7(13)',
    '7(b9)', '7(b13)', '7(#9)', 'aug(7)', 'aug(b7)', '1', '5', 'sus4', 'sus2', '?',
]
REHEARSAL_MARK_MAP = [
    'Intro', 'Ending', 'Fill-in', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M'
]


# noinspection PyMissingConstructor
class XFMidi(pretty_midi.PrettyMIDI):

    def __init__(self, midi_file: str, constant_tempo: Union[None, float] = None, verbose: bool = False):
        """Initialize either by populating it with MIDI data from a file or
        from scratch with no data.

        """
        # Load in the MIDI data using the midi module
        if isinstance(midi_file, six.string_types):
            # If a string was given, pass it as the string filename
            midi_data = mido.MidiFile(filename=midi_file, clip=True)
        else:
            # Otherwise, try passing it in as a file pointer
            midi_data = mido.MidiFile(file=midi_file, clip=True)

        # Convert tick values in midi_data to absolute, a useful thing.
        for track in midi_data.tracks:
            tick = 0
            for event in track:
                event.time += tick
                tick = event.time

        # Store the resolution for later use
        self.resolution = midi_data.ticks_per_beat

        # Populate the list of tempo changes (tick scales)
        if constant_tempo is not None:
            self._tick_scales = [(0, 60.0/(float(constant_tempo)*self.resolution))]
        else:
            self._load_tempo_changes(midi_data)

        # Update the array which maps ticks to time
        max_tick = max([max([e.time for e in t])
                        for t in midi_data.tracks]) + 1
        # If max_tick is huge, the MIDI file is probably corrupt
        # and creating the __tick_to_time array will thrash memory
        if max_tick > pretty_midi.MAX_TICK:
            raise ValueError(('MIDI file has a largest tick of {},'
                              ' it is likely corrupt'.format(max_tick)))

        # Create list that maps ticks to time in seconds
        self._update_tick_to_time(max_tick)

        # Populate the list of key and time signature changes
        self._load_metadata(midi_data)

        # Check that there are tempo, key and time change events
        # only on track 0
        if constant_tempo is None and any(e.type in ('set_tempo', 'key_signature', 'time_signature')
               for track in midi_data.tracks[1:] for e in track):
            warnings.warn(
                "Tempo, Key or Time signature change events found on "
                "non-zero tracks.  This is not a valid type 0 or type 1 "
                "MIDI file.  Tempo, Key or Time Signature may be wrong.",
                RuntimeWarning)

        # Populate the list of instruments
        self._load_instruments(midi_data)

        self.chords = []
        self.rehearsal = []
        self.other_metadata = []
        self.xf_karaoke_message_flag = False
        self.xf_style_message_flag = False
        self.lyric_meta_event_flag = False
        self.xf_information_header_flag = False
        self.xf_version = ''


        for track in midi_data.tracks:
            for event in track:
                if event.type == 'sequencer_specific':
                    data = event.data
                    processed = False
                    if len(data) >= 2 and data[0] == 0x43 and data[1] == 0x7b:
                        annotation_type = data[2]
                        if annotation_type == 0x00:  # xf version id
                            assert len(data) == 9
                            self.xf_version = ''.join(chr(data[i]) for i in range(3, 7))
                            status = data[7] << 8 | data[8]
                            self.xf_information_header_flag = (status & 1 << 0) != 0
                            self.xf_style_message_flag = (status & 1 << 1) != 0
                            assert (status & 1 << 2) == 0, 'unknown flag in xf version ID'
                            self.lyric_meta_event_flag = (status & 1 << 3) != 0
                            self.xf_karaoke_message_flag = (status & 1 << 4) != 0
                            assert (status & 0xffe0) == 0, 'unknown flag in xf version ID'
                            processed = True
                        if annotation_type == 0x01:  # chords
                            assert len(data) == 7

                            def parse_chord_note(code):
                                if code == 0x7f:
                                    return 'N'
                                inversion_number = code >> 4
                                note = code & 0xf
                                assert 0 < note <= 7, 'unknown note %d' % note
                                assert 0 <= inversion_number < 7, 'unknown inversion number %d' % inversion_number
                                return 'XCDEFGAB'[note] + ['bbb', 'bb', 'b', '', '#', '##', '###'][inversion_number]
                            def get_interval(note1, note2):
                                # note1, note2 are formatted as 'C', 'C#', 'D', etc.
                                scale1 = 'C.D.EF.G.A.B'.index(note1[0]) + note1.count('#') - note1.count('b')
                                scale2 = 'C.D.EF.G.A.B'.index(note2[0]) + note2.count('#') - note2.count('b')
                                interval = (scale2 - scale1) % 12
                                return ['1', 'b2', '2', 'b3', '3', '4', 'b5', '5', '#5', '6', 'b7', '7'][interval]


                            def parse_chord_type(code):
                                if code == 0x7f:
                                    return 'N'
                                assert 0 <= code < len(CHORD_MAP), 'unknown chord quality %d' % code
                                return CHORD_MAP[code]
                            chord_symbol = '%s:%s' % (parse_chord_note(data[3]), parse_chord_type(data[4]))
                            if data[5] != 0x7f:
                                # Ignoring secondary chord type data[6]
                                chord_symbol = '%s/%s' % (chord_symbol, get_interval(parse_chord_note(data[3]), parse_chord_note(data[5])))
                            self.chords.append([self._PrettyMIDI__tick_to_time[event.time], chord_symbol])
                            processed = True
                        elif annotation_type == 0x02:  # rehearsal mark
                            assert len(data) == 4
                            r = data[3]
                            assert (r & 0x80) == 0, 'Bad rehearsal annotation'
                            segment_label = REHEARSAL_MARK_MAP[r & 0xf]
                            variation_label = '\'' * (r >> 4)
                            self.rehearsal.append([self._PrettyMIDI__tick_to_time[event.time], segment_label + variation_label])
                            processed = True
                        elif annotation_type == 0x10:  # guitar information flags
                            self.other_metadata.append(event)
                            processed = True
                        if not processed and verbose > 0:
                            print('Warning: annotation not processed:', event.hex())


    def _load_instruments(self, midi_data):
        """Populates ``self.instruments`` using ``midi_data``.

        Parameters
        ----------
        midi_data : midi.FileReader
            MIDI object from which data will be read.
        """
        # MIDI files can contain a collection of tracks; each track can have
        # events occuring on one of sixteen channels, and events can correspond
        # to different instruments according to the most recently occurring
        # program number.  So, we need a way to keep track of which instrument
        # is playing on each track on each channel.  This dict will map from
        # program number, drum/not drum, channel, and track index to instrument
        # indices, which we will retrieve/populate using the __get_instrument
        # function below.
        instrument_map = collections.OrderedDict()
        # Store a similar mapping to instruments storing "straggler events",
        # e.g. events which appear before we want to initialize an Instrument
        stragglers = {}
        # This dict will map track indices to any track names encountered
        track_name_map = collections.defaultdict(str)

        def __get_instrument(program, channel, track, create_new):
            """Gets the Instrument corresponding to the given program number,
            drum/non-drum type, channel, and track index.  If no such
            instrument exists, one is created.

            """
            # If we have already created an instrument for this program
            # number/track/channel, return it
            if (program, channel, track) in instrument_map:
                return instrument_map[(program, channel, track)]
            # If there's a straggler instrument for this instrument and we
            # aren't being requested to create a new instrument
            if not create_new and (channel, track) in stragglers:
                return stragglers[(channel, track)]
            # If we are told to, create a new instrument and store it
            if create_new:
                is_drum = (channel == 9)
                instrument = pretty_midi.Instrument(
                    program, is_drum, track_name_map[track_idx])
                instrument.channel = channel
                # If any events appeared for this instrument before now,
                # include them in the new instrument
                if (channel, track) in stragglers:
                    straggler = stragglers[(channel, track)]
                    instrument.control_changes = straggler.control_changes
                    instrument.pitch_bends = straggler.pitch_bends
                # Add the instrument to the instrument map
                instrument_map[(program, channel, track)] = instrument
            # Otherwise, create a "straggler" instrument which holds events
            # which appear before we actually want to create a proper new
            # instrument
            else:
                # Create a "straggler" instrument
                instrument = pretty_midi.Instrument(program, track_name_map[track_idx])
                # Note that stragglers ignores program number, because we want
                # to store all events on a track which appear before the first
                # note-on, regardless of program
                stragglers[(channel, track)] = instrument
            return instrument

        for track_idx, track in enumerate(midi_data.tracks):
            # Keep track of last note on location:
            # key = (instrument, note),
            # value = (note-on tick, velocity)
            last_note_on = collections.defaultdict(list)
            # Keep track of which instrument is playing in each channel
            # initialize to program 0 for all channels
            current_instrument = np.zeros(16, dtype=np.int32)
            for event in track:
                # Look for track name events
                if event.type == 'track_name':
                    # Set the track name for the current track
                    track_name_map[track_idx] = event.name
                # Look for program change events
                if event.type == 'program_change':
                    # Update the instrument for this channel
                    current_instrument[event.channel] = event.program
                # Note ons are note on events with velocity > 0
                elif event.type == 'note_on' and event.velocity > 0:
                    # Store this as the last note-on location
                    note_on_index = (event.channel, event.note)
                    last_note_on[note_on_index].append((
                        event.time, event.velocity))
                # Note offs can also be note on events with 0 velocity
                elif event.type == 'note_off' or (event.type == 'note_on' and
                                                  event.velocity == 0):
                    # Check that a note-on exists (ignore spurious note-offs)
                    key = (event.channel, event.note)
                    if key in last_note_on:
                        # Get the start/stop times and velocity of every note
                        # which was turned on with this instrument/drum/pitch.
                        # One note-off may close multiple note-on events from
                        # previous ticks. In case there's a note-off and then
                        # note-on at the same tick we keep the open note from
                        # this tick.
                        end_tick = event.time
                        open_notes = last_note_on[key]

                        notes_to_close = [
                            (start_tick, velocity)
                            for start_tick, velocity in open_notes
                            if start_tick != end_tick]
                        notes_to_keep = [
                            (start_tick, velocity)
                            for start_tick, velocity in open_notes
                            if start_tick == end_tick]

                        for start_tick, velocity in notes_to_close:
                            start_time = self._PrettyMIDI__tick_to_time[start_tick]
                            end_time = self._PrettyMIDI__tick_to_time[end_tick]
                            # Create the note event
                            note = pretty_midi.Note(velocity, event.note, start_time,
                                        end_time)
                            # Get the program and drum type for the current
                            # instrument
                            program = current_instrument[event.channel]
                            # Retrieve the Instrument instance for the current
                            # instrument
                            # Create a new instrument if none exists
                            instrument = __get_instrument(
                                program, event.channel, track_idx, 1)
                            # Add the note event
                            instrument.notes.append(note)

                        if len(notes_to_close) > 0 and len(notes_to_keep) > 0:
                            # Note-on on the same tick but we already closed
                            # some previous notes -> it will continue, keep it.
                            last_note_on[key] = notes_to_keep
                        else:
                            # Remove the last note on for this instrument
                            del last_note_on[key]
                # Store pitch bends
                elif event.type == 'pitchwheel':
                    # Create pitch bend class instance
                    bend = pretty_midi.PitchBend(event.pitch,
                                     self._PrettyMIDI__tick_to_time[event.time])
                    # Get the program for the current inst
                    program = current_instrument[event.channel]
                    # Retrieve the Instrument instance for the current inst
                    # Don't create a new instrument if none exists
                    instrument = __get_instrument(
                        program, event.channel, track_idx, 0)
                    # Add the pitch bend event
                    instrument.pitch_bends.append(bend)
                # Store control changes
                elif event.type == 'control_change':
                    control_change = pretty_midi.ControlChange(
                        event.control, event.value,
                        self._PrettyMIDI__tick_to_time[event.time])
                    # Get the program for the current inst
                    program = current_instrument[event.channel]
                    # Retrieve the Instrument instance for the current inst
                    # Don't create a new instrument if none exists
                    instrument = __get_instrument(
                        program, event.channel, track_idx, 0)
                    # Add the control change event
                    instrument.control_changes.append(control_change)
        # Initialize list of instruments from instrument_map
        self.instruments = [i for i in instrument_map.values()]

