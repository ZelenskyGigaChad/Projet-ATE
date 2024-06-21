import time
from dataclasses import dataclass

import pyaudio
import argparse
import asyncio
import aiohttp
import json
import os
import sys
import wave
import websockets

from datetime import datetime
audio_queue = asyncio.Queue()
startTime = datetime.now()

all_mic_data = []
all_transcripts = []

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 16000#8000



# Mimic sending a real-time stream by sending this many seconds of audio at a time.
# Used for file "streaming" only.
REALTIME_RESOLUTION = 0.250

subtitle_line_counter = 0


def subtitle_time_formatter(seconds, separator):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}{separator}{millis:03}"


def subtitle_formatter(response, format):
    global subtitle_line_counter
    subtitle_line_counter += 1

    start = response["start"]
    end = start + response["duration"]
    transcript = response.get("channel", {}).get("alternatives", [{}])[0].get("transcript", "")

    separator = "," if format == "srt" else '.'
    prefix = "- " if format == "vtt" else ""
    subtitle_string = (
        f"{subtitle_line_counter}\n"
        f"{subtitle_time_formatter(start, separator)} --> "
        f"{subtitle_time_formatter(end, separator)}\n"
        f"{prefix}{transcript}\n\n"
    )

    return subtitle_string


# Used for microphone streaming only.
def mic_callback(input_data, frame_count, time_info, status_flag):
    audio_queue.put_nowait(input_data)
    return (input_data, pyaudio.paContinue)


async def run(key, method, format, **kwargs):
    deepgram_url = f'{kwargs["host"]}/v1/listen?punctuate=true&language={ApiData.language}&endpointing=100&interim_result=true'

    if kwargs["model"]:
        deepgram_url += f"&model={kwargs['model']}"

    if kwargs["tier"]:
        deepgram_url += f"&tier={kwargs['tier']}"

    if method == "mic":
        deepgram_url += f"&encoding=linear16&sample_rate={RATE}"

    elif method == "wav":
        data = kwargs["data"]
        deepgram_url += f'&channels={kwargs["channels"]}&sample_rate={kwargs["sample_rate"]}&encoding=linear16'


    # Connect to the real-time streaming endpoint, attaching our credentials.
    async with websockets.connect(
        deepgram_url, extra_headers={"Authorization": "Token {}".format(key)}
    ) as ws:
        print(f'ℹ️  Request ID: {ws.response_headers.get("dg-request-id")}')
        if kwargs["model"]:
            print(f'ℹ️  Model: {kwargs["model"]}')
        if kwargs["tier"]:
            print(f'ℹ️  Tier: {kwargs["tier"]}')
        print("🟢 (1/5) Successfully opened Deepgram streaming connection")

        async def sender(ws):
            global audio_queue
            print(
                f'🟢 (2/5) Ready to stream {method if (method == "mic" or method == "url") else kwargs["filepath"]} audio to Deepgram{". Speak into your microphone to transcribe." if method == "mic" else ""}'
            )

            if method == "mic":
                try:
                    while True:
                        mic_data = await audio_queue.get()
                        all_mic_data.append(mic_data)
                        await ws.send(mic_data)
                except websockets.exceptions.ConnectionClosedOK:
                    await ws.send(json.dumps({"type": "CloseStream"}))
                    print(
                        "🟢 (5/5) Successfully closed Deepgram connection, waiting for final transcripts if necessary"
                    )

            elif method == "url":
                # Listen for the connection to open and send streaming audio from the URL to Deepgram
                async with aiohttp.ClientSession() as session:
                    async with session.get(kwargs["url"]) as audio:
                        while True:
                            remote_url_data = await audio.content.readany()
                            await ws.send(remote_url_data)

                            # If no data is being sent from the live stream, then break out of the loop.
                            if not remote_url_data:
                                break

            elif method == "wav":
                nonlocal data
                # How many bytes are contained in one second of audio?
                byte_rate = (
                    kwargs["sample_width"] * kwargs["sample_rate"] * kwargs["channels"]
                )
                # How many bytes are in `REALTIME_RESOLUTION` seconds of audio?
                chunk_size = int(byte_rate * REALTIME_RESOLUTION)

                try:
                    while len(data):
                        chunk, data = data[:chunk_size], data[chunk_size:]
                        # Mimic real-time by waiting `REALTIME_RESOLUTION` seconds
                        # before the next packet.
                        await asyncio.sleep(REALTIME_RESOLUTION)
                        # Send the data
                        await ws.send(chunk)

                    await ws.send(json.dumps({"type": "CloseStream"}))
                    print(
                        "🟢 (5/5) Successfully closed Deepgram connection, waiting for final transcripts if necessary"
                    )
                except Exception as e:
                    print(f"🔴 ERROR: Something happened while sending, {e}")
                    raise e

            return

        async def receiver(ws):
            """Print out the messages received from the server."""
            first_message = True
            first_transcript = True
            transcript = ""

            async for msg in ws:
                res = json.loads(msg)
                if first_message:
                    print(
                        "🟢 (3/5) Successfully receiving Deepgram messages, waiting for finalized transcription..."
                    )
                    first_message = False
                try:
                    # handle local server messages
                    if res.get("msg"):
                        print(res["msg"])
                    if res.get("is_final"):
                        transcript = (
                            res.get("channel", {})
                            .get("alternatives", [{}])[0]
                            .get("transcript", "")
                        )
                        if kwargs["timestamps"]:
                            words = res.get("channel", {}).get("alternatives", [{}])[0].get("words", [])
                            start = words[0]["start"] if words else None
                            end = words[-1]["end"] if words else None
                            transcript += " [{} - {}]".format(start, end) if (start and end) else ""
                        if transcript != "":
                            if first_transcript:
                                print("🟢 (4/5) Began receiving transcription")
                                # if using webvtt, print out header
                                if format == "vtt":
                                    print("WEBVTT\n")
                                first_transcript = False
                            if format == "vtt" or format == "srt":
                                transcript = subtitle_formatter(res, format)

                            print(f"{datetime.now()}: {transcript}")
                            all_transcripts.append(transcript)

                            f = open("texte.txt", "w", encoding='utf-8')
                            f.write(transcript)
                            f.close()

                        # if using the microphone, close stream if user says "goodbye"
                        if method == "mic" and "goodbye" in transcript.lower():
                            await ws.send(json.dumps({"type": "CloseStream"}))
                            print(
                                "🟢 (5/5) Successfully closed Deepgram connection, waiting for final transcripts if necessary"
                            )

                    # handle end of stream
                    if res.get("created"):
                        # save subtitle data if specified
                        if format == "vtt" or format == "srt":
                            data_dir = os.path.abspath(
                                os.path.join(os.path.curdir, "data")
                            )
                            if not os.path.exists(data_dir):
                                os.makedirs(data_dir)

                            transcript_file_path = os.path.abspath(
                                os.path.join(
                                    data_dir,
                                    f"{startTime.strftime('%Y%m%d%H%M')}.{format}",
                                )
                            )
                            with open(transcript_file_path, "w") as f:
                                f.write("".join(all_transcripts))
                            print(f"🟢 Subtitles saved to {transcript_file_path}")

                            # also save mic data if we were live streaming audio
                            # otherwise the wav file will already be saved to disk
                            if method == "mic":
                                wave_file_path = os.path.abspath(
                                    os.path.join(
                                        data_dir,
                                        f"{startTime.strftime('%Y%m%d%H%M')}.wav",
                                    )
                                )
                                wave_file = wave.open(wave_file_path, "wb")
                                wave_file.setnchannels(CHANNELS)
                                wave_file.setsampwidth(SAMPLE_SIZE)
                                wave_file.setframerate(RATE)
                                wave_file.writeframes(b"".join(all_mic_data))
                                wave_file.close()
                                print(f"🟢 Mic audio saved to {wave_file_path}")

                        print(
                            f'🟢 Request finished with a duration of {res["duration"]} seconds. Exiting!'
                        )
                except KeyError:
                    print(f"🔴 ERROR: Received unexpected API response! {msg}")

        # Set up microphone if streaming from mic
        async def microphone():
            audio = pyaudio.PyAudio()
            stream = audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=mic_callback,
            )

            stream.start_stream()

            global SAMPLE_SIZE
            SAMPLE_SIZE = audio.get_sample_size(FORMAT)

            while stream.is_active():
                await asyncio.sleep(0.1)

            stream.stop_stream()
            stream.close()

        functions = [
            asyncio.ensure_future(sender(ws)),
            asyncio.ensure_future(receiver(ws)),
        ]

        if method == "mic":
            functions.append(asyncio.ensure_future(microphone()))

        await asyncio.gather(*functions)


def validate_input(input):
    if input.lower().startswith("mic"):
        return input

    elif input.lower().endswith("wav"):
        if os.path.exists(input):
            return input

    elif input.lower().startswith("http"):
        return input

    raise argparse.ArgumentTypeError(
        f'{input} is an invalid input. Please enter the path to a WAV file, a valid stream URL, or "mic" to stream from your microphone.'
    )


def validate_format(format):
    if (
        format.lower() == ("text")
        or format.lower() == ("vtt")
        or format.lower() == ("srt")
    ):
        return format

    raise argparse.ArgumentTypeError(
        f'{format} is invalid. Please enter "text", "vtt", or "srt".'
    )

def validate_dg_host(dg_host):
    if (
        # Check that the host is a websocket URL
        dg_host.startswith("wss://")
        or dg_host.startswith("ws://")
    ):
        # Trim trailing slash if necessary
        if dg_host[-1] == '/':
            return dg_host[:-1]
        return dg_host 

    raise argparse.ArgumentTypeError(
            f'{dg_host} is invalid. Please provide a WebSocket URL in the format "{{wss|ws}}://hostname[:port]".'
    )


def main():
    """Entrypoint for the example."""
    # Parse the command-line arguments.
    input = ApiData.input
    format = ApiData.format.lower()
    host = ApiData.host

    try:
        if input.lower().startswith("mic"):
            asyncio.run(run(ApiData.key, "mic", format, model=ApiData.model, tier=ApiData.tier, host=host, timestamps=ApiData.timestamps))

        elif input.lower().endswith("wav"):
            if os.path.exists(input):
                # Open the audio file.
                with wave.open(input, "rb") as fh:
                    (
                        channels,
                        sample_width,
                        sample_rate,
                        num_samples,
                        _,
                        _,
                    ) = fh.getparams()
                    assert sample_width == 2, "WAV data must be 16-bit."
                    data = fh.readframes(num_samples)
                    asyncio.run(
                        run(
                            ApiData.key,
                            "wav",
                            format,
                            model=ApiData.model,
                            tier=ApiData.tier,
                            data=data,
                            channels=channels,
                            sample_width=sample_width,
                            sample_rate=sample_rate,
                            filepath=ApiData.input,
                            host=host,
                            timestamps=ApiData.timestamps,
                        )
                    )
            else:
                raise argparse.ArgumentTypeError(
                    f"🔴 {ApiData.input} is not a valid WAV file."
                )

        elif input.lower().startswith("http"):
            asyncio.run(run(ApiData.key, "url", format, model=ApiData.model, tier=ApiData.tier, url=input, host=host, timestamps=ApiData.timestamps))

        else:
            raise argparse.ArgumentTypeError(
                f'🔴 {input} is an invalid input. Please enter the path to a WAV file, a valid stream URL, or "mic" to stream from your microphone.'
            )

    except websockets.exceptions.InvalidStatusCode as e:
        print(f'🔴 ERROR: Could not connect to Deepgram! {e.headers.get("dg-error")}')
        print(
            f'🔴 Please contact Deepgram Support (developers@deepgram.com) with request ID {e.headers.get("dg-request-id")}'
        )
        return
    except websockets.exceptions.ConnectionClosedError as e:
        error_description = f"Unknown websocket error."
        print(
            f"🔴 ERROR: Deepgram connection unexpectedly closed with code {e.code} and payload {e.reason}"
        )

        if e.reason == "DATA-0000":
            error_description = "The payload cannot be decoded as audio. It is either not audio data or is a codec unsupported by Deepgram."
        elif e.reason == "NET-0000":
            error_description = "The service has not transmitted a Text frame to the client within the timeout window. This may indicate an issue internally in Deepgram's systems or could be due to Deepgram not receiving enough audio data to transcribe a frame."
        elif e.reason == "NET-0001":
            error_description = "The service has not received a Binary frame from the client within the timeout window. This may indicate an internal issue in Deepgram's systems, the client's systems, or the network connecting them."

        print(f"🔴 {error_description}")
        # TODO: update with link to streaming troubleshooting page once available
        # print(f'🔴 Refer to our troubleshooting suggestions: ')
        print(
            f"🔴 Please contact Deepgram Support (developers@deepgram.com) with the request ID listed above."
        )
        return

    except websockets.exceptions.ConnectionClosedOK:
        return



@dataclass
class ApiData:
    key='557089d4de3e8902a5381be50efd9197f0e7263c'
    input='mic'
    model = 'enhanced'
    language="fr"
    tier = ''
    timestamps = False
    format = 'text'
    host = 'wss://api.deepgram.com'

if __name__ == "__main__":

    sys.exit(main() or 0)