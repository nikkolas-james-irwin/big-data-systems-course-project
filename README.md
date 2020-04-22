# Containerized Amazon Recommender System (CARS)

* CARS is an academic project that creates a recommender system using Apache Spark's PySpark API.
* The recommender system is implemented using the [ALS algorithm](https://spark.apache.org/docs/latest/mllib-collaborative-filtering.html) found in the MlLib library.
* The ALS algorithm is used to provide recommendations based on the Amazon 2018 dataset provided by UCSD.
* For development/testing we used the 5-core (14.3gb) - subset of the data in which all users and items have at least 5 reviews (75.26 million total reviews across all datasets).

   ![5-core](https://i.imgur.com/uYGiuyn.png)

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
   * Example data and it's structure from the **Video_Games_5.json** dataset is shown below.
   * **Only the first 10 lines of this dataset are shown, the entire file has 497,577 reviews**.

    ```json
   {"overall": 5.0, "verified": true, "reviewTime": "10 17, 2015", "reviewerID": "A1HP7NVNPFMA4N", "asin": "0700026657", "reviewerName": "Ambrosia075", "reviewText": "This game is a bit hard to get the hang of, but when you do it's great.", "summary": "but when you do it's great.", "unixReviewTime": 1445040000}
    {"overall": 4.0, "verified": false, "reviewTime": "07 27, 2015", "reviewerID": "A1JGAP0185YJI6", "asin": "0700026657", "reviewerName": "travis", "reviewText": "I played it a while but it was alright. The steam was a bit of trouble. The more they move these game to steam the more of a hard time I have activating and playing a game. But in spite of that it was fun, I liked it. Now I am looking forward to anno 2205 I really want to play my way to the moon.", "summary": "But in spite of that it was fun, I liked it", "unixReviewTime": 1437955200}
    {"overall": 3.0, "verified": true, "reviewTime": "02 23, 2015", "reviewerID": "A1YJWEXHQBWK2B", "asin": "0700026657", "reviewerName": "Vincent G. Mezera", "reviewText": "ok game.", "summary": "Three Stars", "unixReviewTime": 1424649600}
    {"overall": 2.0, "verified": true, "reviewTime": "02 20, 2015", "reviewerID": "A2204E1TH211HT", "asin": "0700026657", "reviewerName": "Grandma KR", "reviewText": "found the game a bit too complicated, not what I expected after having played 1602, 1503, and 1701", "summary": "Two Stars", "unixReviewTime": 1424390400}
    {"overall": 5.0, "verified": true, "reviewTime": "12 25, 2014", "reviewerID": "A2RF5B5H74JLPE", "asin": "0700026657", "reviewerName": "jon", "reviewText": "great game, I love it and have played it since its arrived", "summary": "love this game", "unixReviewTime": 1419465600}
    {"overall": 4.0, "verified": true, "reviewTime": "11 13, 2014", "reviewerID": "A11V6ZJ2FVQY1D", "asin": "0700026657", "reviewerName": "IBRAHIM ALBADI", "reviewText": "i liked a lot some time that i haven't play a wonderfull game very simply and funny game verry good game.", "summary": "Anno 2070", "unixReviewTime": 1415836800}
    {"overall": 1.0, "verified": false, "reviewTime": "08 2, 2014", "reviewerID": "A1KXJ1ELZIU05C", "asin": "0700026657", "reviewerName": "Creation27", "reviewText": "I'm an avid gamer, but Anno 2070 is an INSULT to gaming.  It is so buggy and half-finished that the first campaign doesn't even work properly and the DRM is INCREDIBLY frustrating to deal with.\n\nOnce you manage to work your way past the massive amounts of bugs and get through the DRM, HOURS later you finally figure out that the game has no real tutorial, so you stuck just clicking around randomly.\n\nSad, sad, sad, example of a game that could have been great but FTW.", "summary": "Avoid This Game - Filled with Bugs", "unixReviewTime": 1406937600}
    {"overall": 5.0, "verified": true, "reviewTime": "03 3, 2014", "reviewerID": "A1WK5I4874S3O2", "asin": "0700026657", "reviewerName": "WhiteSkull", "reviewText": "I bought this game thinking it would be pretty cool and that i might play it for a  week or two and be done.  Boy was I wrong! From the moment I finally got the gamed Fired up (the other commentors on this are right, it takes forever and u are forced to create an account) I watched as it booted up I could tell right off the bat that ALOT of thought went into making this game. If you have ever played Sim city, then this game is a must try as you will easily navigate thru it and its multi layers. I have been playing htis now for a month straight, and I am STILL discovering layers of complexity in the game. There are a few things in the game that could used tweaked, but all in all this is a 5 star game.", "summary": "A very good game balance of skill with depth of choices", "unixReviewTime": 1393804800}
    {"overall": 5.0, "verified": true, "reviewTime": "02 21, 2014", "reviewerID": "AV969NA4CBP10", "asin": "0700026657", "reviewerName": "Travis B. Moore", "reviewText": "I have played the old anno 1701 AND 1503.  this game looks great but is more complex than the previous versions of the game. I found a lot of things lacking such as the sources of power and an inability to store energy with batteries or regenertive fuel cells as buildings in the game need power. Trade is about the same. My main beef with this it requires an internet connection. Other than that it has wonderful artistry and graphics. It is the same as anno 1701 but set in a future world where global warmming as flood the land and resource scarcity has sent human kind to look to the deep ocean for valuable minerals. I recoment the deep ocean expansion or complete if you get this. I found the ai instructor a little corny but other than that the game has some real polish. I wrote my 2 cents worth on suggestions on anno 2070 wiki and you can read 3 pages on that for game ideas I had.", "summary": "Anno 2070 more like anno 1701", "unixReviewTime": 1392940800}
    {"overall": 4.0, "verified": true, "reviewTime": "06 27, 2013", "reviewerID": "A1EO9BFUHTGWKZ", "asin": "0700026657", "reviewerName": "johnnyz3", "reviewText": "I liked it and had fun with it, played for a while and got my money's worth.  You can certainly go further than I did but I got frustrated with the fact that here we are in this new start and still taking from the earth rather than living with it. Better than simcity in that respect and maybe the best we could hope for.", "summary": "Pretty fun", "unixReviewTime": 1372291200}
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
    docker container run -d -p 8888:8888 -p 4040:4040 -it --name cars_container --mount source=cars_local_volume,target=/home/jovyan/work cars
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
