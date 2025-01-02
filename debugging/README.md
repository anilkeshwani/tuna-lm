# Debugging README

## Getting inference working as expected

Minimum _working_ example. This works:

```
tune run generate --config ./working_generation_config.yaml
```

Override the prompt specified in the config:

```
tune run generate --config ./working_generation_config.yaml \
prompt="Here's a funny joke"
```

### Inference with extended base model (no CPT)

I had to decrease temperature to get meaningful output

Using the working_generation_config_extended_base.yaml:

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

## My inference script ("recipe")

Check our inference impl with the OOTB recipe and base model.

```bash
tune run tunalm/modern_inference.py \
    --config /mnt/scratch-artemis/anilkeshwani/tmp/debug_tt_inference/custom_generation_config.yaml \
    prompt="What are some interesting sites to visit in the Bay Area?"
```
