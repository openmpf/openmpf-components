# Overview
This repository contains source code for the OpenMPF Azure Cognitive Services Speech-to-Text Component. This component utilizes the [Azure Cognitive Services Batch Transcription REST endpoint](https://docs.microsoft.com/en-us/azure/cognitive-services/speech-service/batch-transcription) to transcribe speech from audio and video files.


# Required Job Properties
In order for the component to process any jobs, the job properties listed below must be provided. These properties have no default value, but can be set through environment variables of the same name. If both environment variable and job property are provided, the job property will be used.

- `ACS_URL`: URL for the Azure Cognitive Services Endpoint.
 e.g. `https://virginia.cris.azure.us/api/speechtotext/v2.0/transcriptions`.
 The component has only been tested against v2.0 of the API.

 - `ACS_SUBSCRIPTION_KEY`: A string containing your subscription key for the speech service.

- `ACS_BLOB_CONTAINER_URL`: URL for an Azure Storage Blob container in which to store files during processing.
 e.g. `https://myaccount.blob.core.windows.net/mycontainer`.
 See Microsoft's [documentation on Azure storage](https://docs.microsoft.com/en-us/azure/storage/blobs/storage-blob-container-create) for details.

- `ACS_BLOB_SERVICE_KEY`: A string containing your Azure Cognitive Services storage access key.


# Optional Job Properties
The below properties can be optionally provided to alter the behavior of the component.

- `LANGUAGE`:  The locale to use for transcription. Defaults to `en-US`. A complete list of available locales can be found in Microsoft's [Speech service documentation](https://docs.microsoft.com/en-us/azure/cognitive-services/speech-service/language-support).

- `DIARIZE`: Whether to assign utterances to different speakers. Currently, this component supports only two-speaker diarization. Diarization is enabled by default.

- `CLEANUP`: Whether to delete files from Azure Blob storage container when processing is complete. It is recommended to always keep this enabled, unless it is expected that the same piece of media will be processed multiple times.

- `BLOB_ACCESS_TIME`: The amount of time in minutes for which the Azure Speech service will have access to the file in blob storage.

# Sample Program
`sample_acs_speech_detector.py` can be used to quickly test with the Azure endpoint. Run with the `-h` flag to see accepted command-line arguments.
