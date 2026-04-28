
Getting Started
===============
This page documents how to install the Panel Segmentation package and run 
automated metadata extraction for a PV array at a specified location. 
These instructions assume that you already have Anaconda and git installed. 
This page also documents how to set up a Google Maps Static API key, which is needed when generating satellite imagery.


Installation Guide
------------------
To install Panel-Segmentation, perform the following steps:

1. Enable Git large file storage (lfs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You must have Git large file storage (lfs) on your computer in order to download the deep learning models in this package.
Go to the following site to download Git lfs: 

.. code-block:: console

    https://git-lfs.github.com/

2. Pip install panel-segmentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Once git lfs is installed, you can now install Panel-Segmentation on your computer.
We are still working on making panel-segmentation available via PyPi, so entering the following in the command line will install the package locally on your computer:

.. code-block:: console

    pip install git+https://github.com/NREL/Panel-Segmentation.git@master#egg=panel-segmentation

3. Pip install mmcv
^^^^^^^^^^^^^^^^^^^
Panel-Segmentation requires the MMCV package, which can be tricky to install for CPU-only, and needs to be installed from source.
To install MMCV for source, run the following in the command line:

.. code-block:: console

    pip install git+https://github.com/open-mmlab/mmcv.git@v2.1.0

4. Set model paths
^^^^^^^^^^^^^^^^^^
When initiating the PanelDetection() class, be sure to point your file paths to the model paths in your local Panel-Segmentation folder!

Please note that installations will likely take several minutes.


.. _google-api-key-setup:
Setting Up Google Maps Static API Key
-------------------------------------
The Google Maps Static API allows you to pull satellite imagery from Google Maps given a set of latitude-longitude coordinates.
The following link describes how to set up a Google Maps Static API key:

.. code-block:: console

    https://developers.google.com/maps/documentation/maps-static/get-api-key

For reference, you can also perform the following steps:

1. Review Maps Static API costs.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Before using the Maps Static API, you should review the costs for using the Maps Static API at https://developers.google.com/maps/billing-and-pricing/pricing.
The following table shows the cost for using the Maps Static API as of September 2025.

+-------------+----------------+--------------+---------------+-------------------+---------------------+------------+
| API Queries | Free Usage Cap | Cap-100,000  | 100,001-500K  | 500,001-1,000,000 | 1,000,001-5,000,000 | 5,000,000+ |
+=============+================+==============+===============+===================+=====================+============+
| Static Maps | 10,000         | $2.00        | $1.60         | $1.20             | $0.60               | $0.15      |
+-------------+----------------+--------------+---------------+-------------------+---------------------+------------+

There is a free usage cap of 10,000 API queries per month and this refreshes every month.
You should also limit pulling thousands of images in one setting to avoid being flagged as webscraping by Google. 
To avoid this issue, you should pull less than 6,000 images in one day.

2. Sign into Google Cloud.
^^^^^^^^^^^^^^^^^^^^^^^^^^
Create an account or sign into an existing account on Google Cloud at:
The following link describes how to set up a Google Maps Geocoding API key:

.. code-block:: console
    
    https://console.cloud.google.com/.

3. Create or select a Google Cloud Project. 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Create a new project or select an existing project.
First, go to the Google Cloud Console https://console.cloud.google.com/.
At top left corner of the console and on the right of the Google Cloud logo, click on the "project picker" tab. 
You can also use Ctrl O keyboard shortcut on Chrome to open the "project picker" tab.
This opens a project menu and users can select an existing project from the menu or create a New Project by selecting "New Project" on the top left corner.

4. Enable Billing on the account and project.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
On the top left corner of the console, click on the ≡ (hamburger menu) icon.
Select "Billing" from the sidebar menu and enter the billing information that will be used for the project.

5. Enable Maps Static API on the project.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
On the top left corner of the console, click on the ≡ (hamburger menu) icon.
Hover over "API & Services" from the sidebar menu and select "Enable APIs and Services".
From the search bar, search for "Maps Static API" and click on the Maps Static API from the search results.
Click "enable" to enable the Maps Static API.

6. Get API Key.
^^^^^^^^^^^^^^^
On the top left corner of the console, click on the ≡ (hamburger menu) icon.
Hover over "API & Services" from the sidebar menu and select "Credentials".
Under the "API Keys" section, there is an API key associated with the project the user created or selected earlier.
On the right side of the API Keys section, click on "Show Key" to show the API key.
Copy the API key to use in the PanelDetection() class for when satellite imagery is generated.


Setting Up Google Maps Geocoding API Key
----------------------------------------
The Google Maps Geocoding API allows you to get the address of a place given its latitude-longitude coordinates.
This Geocoding API will be used in the Sol-Searcher pipeline to get the address of a solar panel given its latitude-longitude coordinates.
The geocoding API is more expensive than the Maps Static API.
The following table shows the costs for using the Geocoding API as of September 2025.

+-------------+----------------+--------------+---------------+-------------------+---------------------+------------+
| API Queries | Free Usage Cap | Cap-100,000  | 100,001-500K  | 500,001-1,000,000 | 1,000,001-5,000,000 | 5,000,000+ |
+=============+================+==============+===============+===================+=====================+============+
| Geocoding   | 10,000         | $5.00        | $4.00         | $3.00             | $1.50               | $0.38      |
+-------------+----------------+--------------+---------------+-------------------+---------------------+------------+

The steps for setting up the Geocoding API key are the same as the steps for setting up the Maps Static API key.
The only difference is that you should search for and enable the "Geocoding API" instead of "Maps Static API" when performing step 5.
If you have both Geocoding and Maps Static API enabled under the same project, you will use the same API key.
Otherwise, you can have separate projects for each API and use different API keys.

For more information about setting up the Geocoding API, please go to the following link:

.. code-block:: console
    
    https://developers.google.com/maps/documentation/geocoding/get-api-key
