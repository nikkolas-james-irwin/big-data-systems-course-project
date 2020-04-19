# Containerized Amazon Recommender System (CARS)

* CARS is an academic project that creates a recommender system using Apache Spark's PySpark API.
* The recommender system is implemented using the [ALS algorithm](https://spark.apache.org/docs/latest/mllib-collaborative-filtering.html) found in the MlLib library.
* The ALS algorithm is used to provide recommendations based on the Amazon 2018 dataset provided by UCSD.

## Project Details

### Institution

* **University:** [University of Nevada, Reno](https://www.unr.edu)
* **College:** [College of Engineering](https://www.unr.edu/engineering)
* **Department:** [Computer Science and Engineering Department](https://www.unr.edu/cse)
* **Instructor:** Dr. Feng Yan, [Faculty Webpage](https://www.unr.edu/cse/people/feng-yan)
* **Course:** CS 636 - Big Data Systems
  * This course offers an advanced study of state-of-the-art big data techniques and applications and focuses on the tools and systems for big data analytics.

### Project Goals

Our goals are broken down in the bulleted-list below. This list it not exhaustive, however, it does cover the primary goals of this project.

* **Isolate** our [Python](https://www.python.org) application with [Conda](https://docs.conda.io/projects/conda/en/latest/index.html).
* **Manage** package dependencies with [Conda](https://docs.conda.io/projects/conda/en/latest/index.html).
* **Create** a portable application that can be easily distributed and built by all team members using [Docker](https://www.docker.com).
* **Use** [Spark](https://spark.apache.org), a unified analytics engine for big data to perform processing on large datasets of Amazon review data.
* **Prototype** and collaborate on the applications implementation with [Microsoft Visual Studio Code Live Share](https://code.visualstudio.com/blogs/2017/11/15/live-share) and [Jupyter Notebook](https://jupyter.org).
* **Automate** our container and simplify the building and running of our container with [Ansible](https://www.ansible.com).
* **Provide** modern analytics insights through visually appealing insights with [Matplotlib](https://matplotlib.org) and [Plotly](https://plotly.com).
* **Manage** the project's development and deployment with [Git](https://git-scm.com) and [GitHub](https://github.com/about).

### Team Members

**Responsibilities** - Our team split responsibilities into two main areas described below. While the sections below describe the primary work performed by each team member, we still helped each other where possible across areas as time permitted.

#### PySpark Implementation and Data Visualization

* Adam Cassell, [GitHub](https://github.com/casselldev)
* Andrew Munoz, [GitHub](https://github.com/amunoz247)

#### Conda, Docker, and Ansible Containerization and Configuration Management

* Brianna Blain-Castelli, [GitHub](https://github.com/bblain18)
* Nikkolas Irwin, [GitHub](https://github.com/nikkolas-james-irwin)

### Citation

[Amazon Review Data (2018)](https://nijianmo.github.io/amazon/index.html)

**Justifying recommendations using distantly-labeled reviews and fined-grained aspects**
Jianmo Ni, Jiacheng Li, Julian McAuley
_Empirical Methods in Natural Language Processing (EMNLP), 2019_
[PDF](http://cseweb.ucsd.edu/~jmcauley/pdfs/emnlp19a.pdf)

### Example Usage

Follow the steps **in order** to build the application and deploy the Jupyter Notebook. If followed properly, you will be able to write your own application code in the **_/home/jovyan/work/_** directory in a new Jupyter Notebook file or run our application using whichever dataset(s) you have downloaded to your local machine.

1. Download one or more **review** datasets from here: [Amazon Review Data (2018)](https://nijianmo.github.io/amazon/index.html).
   * Place the review dataset JSON file in the project directory (same directory as the Dockerfile).
   * Example data and it's structure from the **Musical_Instrucments.json** dataset is shown below.
   * **Only the first 10 lines of this dataset are shown, the entire file has 1,512,530 reviews**.

    ```json
    {"overall": 5.0, "vote": "90", "verified": false, "reviewTime": "08 9, 2004", "reviewerID": "AXHY24HWOF184", "asin": "0470536454", "style": {"Format:": " Paperback"}, "reviewerName": "Bendy", "reviewText": "Crocheting for Dummies by Karen Manthey & Susan Brittain is a wonderfully thorough and very informative book for anyone wanting to learn to crochet and or wanting to freshen up their skills.\n\nThe book reads like a storybook in paragraph form.  Everything is explained in great detail from choosing yarns and hooks, to how to work a large array of crochet stitches, to how to read a pattern, right down to how to care for ones crocheted items.\n\nThe stitch drawings are clear and expertly done making learning new stitches so much easier.\n\nThe book has both a contents page and an index for easy referral.  I especially liked the fact that an index was included.  So many crochet books do not include this.  The index makes it very easy to find information on a particular topic quickly.\n\nThe recommendations for people just learning to crochet are fantastic.  This book wasn't out when I learned to crochet and I learned the hard way about many of the pit falls this book helps one to avoid.  For instance they recommend one start out with a size H-8 crochet hook and a light colored worsted weight yarn.  I learned with a B-1 hook and a fingering weight yarn.  After 2 whole days of crocheting it was 36\" long and 1.5\" tall.  I was trying to make a baby blanket for my doll (which never got made).\n\nThe book contains humor, not just in the cartoons but in the instructions as well which makes for very entertaining reading while one learns a new craft.  I always appreciate having a teacher with a sense of humor!\n\nA good sampling of designs is included so that one can try out their skills.  These include sweaters, an afghan, doilies, hot pads, pillow, scarves, floral motifs, and bandanas.\n\nI am a crochet designer and I read the book cover to cover like a storybook while on vacation this past week.  I thoroughly enjoyed it and learned a few things as well.  I would highly recommend this book to anyone interested in the art of crochet.", "summary": "Terrific Book for Learning the Art of Crochet", "unixReviewTime": 1092009600}
    {"overall": 4.0, "vote": "2", "verified": true, "reviewTime": "04 6, 2017", "reviewerID": "A29OWR79AM796H", "asin": "0470536454", "style": {"Format:": " Hardcover"}, "reviewerName": "Amazon Customer", "reviewText": "Very helpful...", "summary": "Four Stars", "unixReviewTime": 1491436800}
    {"overall": 5.0, "verified": true, "reviewTime": "03 14, 2017", "reviewerID": "AUPWU27A7X5F6", "asin": "0470536454", "style": {"Format:": " Paperback"}, "reviewerName": "Amazon Customer", "reviewText": "EASY TO UNDERSTAND AND A PROMPT SERVICE TOO", "summary": "Five Stars", "unixReviewTime": 1489449600}
    {"overall": 4.0, "verified": true, "reviewTime": "02 14, 2017", "reviewerID": "A1N69A47D4JO6K", "asin": "0470536454", "style": {"Format:": " Paperback"}, "reviewerName": "Christopher Burnett", "reviewText": "My girlfriend use quite often", "summary": "Four Stars", "unixReviewTime": 1487030400}
    {"overall": 5.0, "verified": true, "reviewTime": "01 29, 2017", "reviewerID": "AHTIQUMVCGBFJ", "asin": "0470536454", "style": {"Format:": " Paperback"}, "reviewerName": "Amazon Customer", "reviewText": "Arrived as described. Very happy.", "summary": "Very happy.", "unixReviewTime": 1485648000}
    {"overall": 5.0, "verified": true, "reviewTime": "01 4, 2017", "reviewerID": "A1J8LQ7HVLR9GU", "asin": "0470536454", "style": {"Format:": " Kindle Edition"}, "reviewerName": "Iheartmanatees", "reviewText": "Love the Dummies Series.  Never fails.", "summary": "Love the Dummies Series", "unixReviewTime": 1483488000}
    {"overall": 5.0, "verified": true, "reviewTime": "01 2, 2017", "reviewerID": "ABVTZ63S6GOWF", "asin": "0470536454", "style": {"Format:": " Paperback"}, "reviewerName": "D. Eva", "reviewText": "Good book.", "summary": "Five Stars", "unixReviewTime": 1483315200}
    {"overall": 4.0, "verified": true, "reviewTime": "12 21, 2016", "reviewerID": "A2HX9NFBXGSWRW", "asin": "0470536454", "style": {"Format:": " Paperback"}, "reviewerName": "Stoeffels", "reviewText": "Just started reading it. Love the charts & cautions.", "summary": "Clear. Good reminders.", "unixReviewTime": 1482278400}
    {"overall": 4.0, "verified": true, "reviewTime": "12 20, 2016", "reviewerID": "AP1TQR64HQRCI", "asin": "0470536454", "style": {"Format:": " Paperback"}, "reviewerName": "nan ekelund", "reviewText": "GREAT  book", "summary": "Four Stars", "unixReviewTime": 1482192000}
    {"overall": 5.0, "verified": true, "reviewTime": "12 15, 2016", "reviewerID": "A37FC9MED20AO", "asin": "0470536454", "style": {"Format:": " Paperback"}, "reviewerName": "Jacqueline Bryant", "reviewText": "this is a very helpful book.", "summary": "Five Stars", "unixReviewTime": 1481760000}
    ```

2. Build the image with Docker using the Dockerfile.

    ```docker
    docker image build -t cars:latest .
    ```

3. Verify that the image was built successfully.

    ```docker
    docker image ls
    ```

4. Create the Docker container and mount the volume to ensure that work performed is saved regardless of the lifecycle/state of the container.

    ```docker
    docker container run -d -p 8888:8888 -it --name cars_container --mount source=cars_local_volume,target=/home/jovyan/work cars
    ```

5. Verify that the volume was initialized successfully.

    ```docker
    docker volume ls
    ```
6. Verify that the container was created successfully and is currently running.

    ```docker
    docker container ls
    ```

7. Get the URL and token value for the Jupyter Notebook.

    ```docker
    docker container exec -it cars_container jupyter notebook list
    ```

### DockerHub Image

The latest version of the CARS image can be found [here](https://hub.docker.com/repository/docker/nikkirwin/cars). You can pull the image by running the following command.

```docker
docker pull nikkirwin/cars:latest
```
