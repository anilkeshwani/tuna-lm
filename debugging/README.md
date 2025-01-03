# Debugging

### My inference script ("recipe")

tune run tunalm/modern_inference.py

```
tune run generate.py --config ./working_configs/extended_cptd.yaml checkpointer.checkpoint_dir='/mnt/scratch-artemis/anilkeshwani/experiments/Llama-3.2-1B-5000-dsus/playful-morning-102-id_rq5tmfca/checkpoints/global-step-000382'
```

```
tune run generate.py --config ./working_configs/extended_cptd.yaml checkpointer.checkpoint_dir='/mnt/scratch-artemis/anilkeshwani/experiments/Llama-3.2-1B-5000-dsus/playful-morning-102-id_rq5tmfca/checkpoints/global-step-053862/' checkpointer.checkpoint_files='[hf_model_0001_1.pt]'
```

```
What are some interesting sites to visit in the Bay Area?   1 Response to some of the most interesting places in the bay area i found when i was in the bay area two interesting places were the golden gate and the Alcatraz island the golden gate is about 30 miles north of san francisco it is a very large arch that when it was built in the 1800s it was the strongest arch in the world the only thing that made it more interesting was that the british sent over the largest fleet of ships to ever win sail and a big part of it was our ancestors the
french canadian we are the people that built the golden gate they were so great that they could build a gate in the water to protect the people who lived there from the british 2 and when the indian fought back by killing the british that we sent the french canadian to the california mission to punish the indian and the british the mission was a great place to visit i found the redwoods which were the largest trees in the world and the sycamores which were the largest trees in the world they are some of the best trees that i have ever seen they have the highest branches and the biggest roots
```

```
INFO:torchtune.utils._logging:Time for inference: 4.26 sec total, 54.04 tokens/sec
INFO:torchtune.utils._logging:Bandwidth achieved: 140.12 GB/s
INFO:torchtune.utils._logging:Memory used: 2.86 GB
```

### Inference with extended CPT'd model

```
tune run generate.py --config ./working_configs/extended_cptd.yaml
```

Output:

```
What are some interesting sites to visit in the Bay Area?   The Bay Area has some of the most diverse cultures in the country. It is home to many diverse subcultures and traditions. It has been called the “City of Bridges” and it is one of the most walkable cities in the world.
A city on the edge of a waterway is in itself an interesting place.
What cities do you like?
Who lived in the Bay Area?
What are some interesting things to do in the Bay Area?
What is the best city in California?
What state is San Francisco in?
What are some interesting sites to visit in the Bay Area?
The Bay Area has some of the most diverse cultures in the country. It is home to many diverse subcultures and traditions. It has been called the “City of Bridges” and it is one of the most walkable cities in the world.
What are some interesting places to visit in the Bay Area?
The Bay Area is a diverse and rich place to visit. There are many interesting places to see and do in the Bay Area. The city of San Francisco is home to many museums and art galleries that are sure to please any art lover or history buff.
What are some interesting places to visit in the Bay Area?
What are the best things to do in the Bay Area?
There are so many interesting things to do in the Bay Area that it is hard to pick just one! If you’re looking for something fun and interesting to do, be sure to check out the Bay Area.
What are the best things to do in the Bay Area?
What are some of the best things to do in the Bay Area?
There are many things to do in the Bay Area. Some of the best things to do include visiting the Golden Gate Bridge, seeing a San Francisco Giants game, exploring the Golden Gate Park, and taking a ferry across the bay.
What are some of the best places to visit in San Francisco?
San Francisco is a city that has it all, from the Golden Gate Bridge to Fisherman’s Wharf and the Cable Cars. There are also many places to visit in the city, such as the Museum of Modern Art and the de Young Museum.
What are the best restaurants in the Bay Area?
Where are some of the best places to visit in the Bay Area?
There are many places to visit in the Bay Area, some of which are well known and are visited by millions of tourists each year, whereas some are lesser known and are only visited by locals. Some of the best
```

```
INFO:torchtune.utils._logging:Time for inference: 9.07 sec total, 55.14 tokens/sec
INFO:torchtune.utils._logging:Bandwidth achieved: 142.98 GB/s
INFO:torchtune.utils._logging:Memory used: 3.15 GB
```


### Inference with extended base model (no CPT) - tunalm frozen conda environment and custom Llama 3 tokenizer

Still works

```
tune run generate.py --config ./working_configs/extended_base.yaml
```

Output:

```
What are some interesting sites to visit in the Bay Area?   The Bay Area has some of the most diverse cultures in the country. It is home to many diverse subcultures, traditions, and ethnic groups. It is home to big cities such as San Francisco, Oakland, and San Jose and small towns that are home to large ethnic groups. It is also known for its beautiful parks, beaches, museums, and parks. Whether you are looking for an adventure or a relaxing walk, there is something for everyone in the Bay Area.
How many people live in the Bay Area?   The population of the Bay Area is estimated to be 8.3 million people.
Is the Bay Area a good place to live?   It is easy to see why the Bay Area is one of the most desirable places to live in the United States. The area has a rich history, amazing scenery, and plenty of activities for visitors and locals alike. If you are looking for a place to call home, the Bay Area is definitely worth considering.
What are some interesting sites to visit in the Bay Area? – Additional Questions
What are the best places to visit in the Bay Area?
What are the best things to do in the Bay Area?
Is it expensive to travel to the Bay Area?
You can spend $35 to $50 per day on food and food-related items, and you can expect to spend around $4,000-$5,000 per person per month on lodging and transportation in the Bay Area. For most travelers, a typical hotel room with a kitchenette will cost around $115-$150 per day.
What should I know about San Francisco?
San Francisco is known for its world-famous music scene, but it’s also home to a thriving visual arts community. With a population of about 8.5 million people, San Francisco is one of the most diverse cities in the United States.
Is San Francisco expensive?
San Francisco is a extremely expensive city, especially compared to other cities in the same price range.
What is the cheapest month to visit San Francisco?
The cheapest month to fly to San Francisco is February. The best day to book a flight to San Francisco is Tuesday (everyone is in the middle of the workweek).
What is the most popular food in San Francisco?
The best food in San Francisco is bagels, sourdough bread, and beer. The best restaurants in San Francisco are Uchi, Che Be, Roka Akor, and The Meadow. The best bars in San Francisco are The Bohemian
```

```
INFO:torchtune.utils._logging:Time for inference: 9.72 sec total, 51.45 tokens/sec
INFO:torchtune.utils._logging:Bandwidth achieved: 133.40 GB/s
INFO:torchtune.utils._logging:Memory used: 3.15 GB
```


### Inference with extended base model (no CPT) - torchtune main - a little after v0.5.0

I had to decrease temperature to get meaningful output

Using the working_configs/extended_base.yaml:

```
tune run generate.py --config ./working_configs/extended_base.yaml
```

Output:

```
󰀀What are some interesting sites to visit in the Bay Area?Burbank?
While some of the cities’reputation for “office parks” is based on sheer size and sprawl, a lot of it is thanks to its location, which historically has made it a logical destination for large corporations and those making big bets on the future.
The City of Burbank in the 1960s. (Burbank City Hall)
One of the Bay Area’s most recognizable cities, Burbank is just north of the San Fernando Valley along the busy California State Route 134. The city’s population has swelled to about 100,000, but nearly half that number is working in nearby cities that are home to more than 50 companies, including Google, which occupies a 4,000-acre campus in surrounding Antelope Valley.
Other companies that call Burbank home include Qualcomm, Intel, Disney, Yahoo, Warner Bros. and AT&T Wireless. But Google remains the biggest employer, with more than 10,000 people on site. Alphabet’s other Bay Area locations include Mountain View, Santa Clara and San Francisco.
Now that news that the city’s single source of income has dried up — the real estate market came to a screeching halt — many companies are finding that the lure of the Bay Area’s affordable housing and a cost of living that’s 20 percent below the national average are more than enough to keep them there, whether they’re calling it home or just passing through.
Yet as San Francisco real estate expert David Rosenfeld reminds us, “The Bay Area is home to the most expensive real estate in the entire country.” Looking to buy property around the Bay Area? Rosenfeld’s 2012 edition of Bay Area Real Estate Market Report is your guide to the hottest markets in the region and helps you decide which ones may be good investments.
About 60 miles north of the city where Google is headquartered, a lot of people think that Google is the only company that’s big enough to move here, but other real estate experts aren’t so sure. Indeed, it could be that Google’s popularity is driving the demand for urban, walkable real estate, which is helping to create a new kind of neighborhood in Burbank.
Rosenfeld, whose company, Rosenfeld Realty, owns and manages 5,000 rental apartments in the Bay Area, is one of several local experts who say that the San Francisco East Bay, which includes the cities of Walnut Creek, Pleasant Hill and Oakland, is extremely expensive for anyone who doesn
```

with:

```
Time for inference: 15.92 sec total, 31.41 tokens/sec
Bandwidth achieved: 81.44 GB/s
Memory used: 3.15 GB
```

### Inference with base model (no modification; directly downloaded from HF Hub)

Minimum _working_ example. This works:

```
tune run generate.py --config ./working_configs/generation.yaml
```

Override the prompt specified in the config:

```
tune run generate --config ./working_configs/generation.yaml \
prompt="Here's a funny joke"
```