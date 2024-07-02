# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

# Unit tests for BPETokenizer

from pathlib import Path
from testing import *
from builtin._location import __source_location

from pipelines.tokenizer.regex import Match, Regex


def test_email():
    regex = Regex(r"([[:alnum:]_.+-]+)@([[:alnum:]-]+)\.([[:alnum:]-]+)")
    email = "stef@modular.com"
    m = regex.find(email)
    assert_true(m)
    assert_equal(email, str(m.value()))
    assert_equal("stef", str(m.value()[1]))
    assert_equal("modular", str(m.value()[2]))
    assert_equal("com", str(m.value()[3]))

    haystack = "Hi! I'm Stef, my email is stef@modular.com."
    m2 = regex.find(haystack)
    assert_true(m2)
    assert_equal(email, str(m2.value()))
    assert_equal("stef", str(m2.value()[1]))
    assert_equal("modular", str(m2.value()[2]))
    assert_equal("com", str(m2.value()[3]))


alias TEST_STORY = """
Young farm boy luke@rebellion.org discovers a message hidden in a droid belonging
to rebel princess leia@rebellion.org, requesting help from the retired Jedi knight
obiwan@jedi.org. After his family is killed by the Empire, luke@rebellion.org
joins obiwan@jedi.org, a smuggler named han@smugglers.net, and his Wookiee
co-pilot chewbacca@smugglers.net to rescue leia@rebellion.org from the Death
Star, an immense space station commanded by vader@empire.gov. They succeed,
and luke@rebellion.org destroys the Death Star using the Force.
"""


@value
struct Email:
    var name: String
    var domain: String
    var tld: String

    fn __str__(self) -> String:
        return self.name + "@" + self.domain + "." + self.tld


def test_findall():
    var t1 = str("")
    for m in Regex(".").findall("hello"):
        t1 += str(m)
    assert_equal("hello", t1)

    email_regex = Regex(r"([[:alnum:]_.+-]+)@([[:alnum:]-]+)\.([[:alnum:]-]+)")
    expected = List(
        Email("luke", "rebellion", "org"),
        Email("leia", "rebellion", "org"),
        Email("obiwan", "jedi", "org"),
        Email("luke", "rebellion", "org"),
        Email("obiwan", "jedi", "org"),
        Email("han", "smugglers", "net"),
        Email("chewbacca", "smugglers", "net"),
        Email("leia", "rebellion", "org"),
        Email("vader", "empire", "gov"),
        Email("luke", "rebellion", "org"),
    )
    i = 0
    for m in email_regex.findall(TEST_STORY):
        email = expected[i]
        assert_equal(str(email), str(m))
        assert_equal(email.name, str(m[1]))
        assert_equal(email.domain, str(m[2]))
        assert_equal(email.tld, str(m[3]))
        i += 1


def main():
    test_email()
    test_findall()
