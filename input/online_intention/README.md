
# Shopper Intent Prediction from Clickstream E‑Commerce Data with Minimal Browsing Information 
_Public Data Release 1.0.0_


### Overview
This repo contains the description of the data released in conjunction with our _Nature Scientific Reports_ 
paper [Shopper Intent Prediction from Clickstream E‑Commerce Data with Minimal Browsing Information](https://rdcu.be/b8oqN).

### Data Download

The dataset is available for _research and educational_ purposes [here](https://www.coveo.com/en/ailabs/shopper-intent-prediction-from-clickstream-e-commerce-data-with-minimal-browsing-information).
To obtain the dataset, you are required to fill out a form with information about you and your institution, and
agree to the _Terms And Conditions_ for fair usage of the data.

For convenience, _Terms And Conditions_ are also included in a pure `txt` format in this repo: 
usage of the data implies the acceptance of these _Terms And Conditions_.

### Data Structure

The dataset is provided as one big text file (`.csv`), inside a `zip` archive containing an additional copy of the 
_Terms And Conditions_. The final dataset contains 5.433.611 individual events, and it is the first dataset of this
kind to be released to the research community. A sample file is included in this repository, showcasing the data structure. 

Field | Type | Description
------------ | ------------- | -------------
session_id_hash | string | Hashed identifier of the shopping session. A session groups together events that are at most 30 minutes apart: if the same user comes back to the target website after 31 minutes from the last interaction, a new session identifier is assigned.
event_type | enum | The type of event according to the [Google Protocol](https://developers.google.com/analytics/devguides/collection/protocol/v1), one of { _pageview_ , _event_ }; for example, an _add_ event can happen on a page load, or as a stand-alone event.
product_action | enum | One of { _detail_, _add_, _purchase_, _remove_, _click_ }. If the field is empty, the event is a simple page view (e.g. the `FAQ` page) without associated products.
product_skus_hash | string | If the event is a _product_ event, hashed identifiers of all products in the event (e.g. all the products in a transaction), pipe separated.
server_timestamp_epoch_ms | int | Epoch time, in milliseconds. The epoch time has been shifted in time to further anonymize the data.
hashed_url | string | Hashed url of the current web page.


We refer the reader to the original [paper](https://rdcu.be/b8oqN) for an extended explanation of how to use the dataset for the
clickstream prediction challenge. Usage of this data implies the acceptance of the _Terms And Conditions_ as set forward in
the [download page](https://www.coveo.com/en/ailabs/shopper-intent-prediction-from-clickstream-e-commerce-data-with-minimal-browsing-information).


